"""Multi-GPU, multi-backbone evaluator for LIBERO, RoboTwin, and RoboDojo.

Input: a model reference — a local VLA checkpoint dir, a Hugging Face repo id
(`user/repo`), or a HF URL (`https://huggingface.co/user/repo`).

What it does:
  1. Resolves the model to a local checkpoint directory. HF references are
     downloaded pinned to --commit-id (mandatory), and the download cache is
     keyed by that commit — a re-submission to the same repo can never be
     confused with a previously evaluated version.
  2. Prepares the selected benchmark. LIBERO-family evaluations use the local
     client and selected policy server; RoboTwin and RoboDojo delegate simulation
     to pinned compatible upstream checkouts.
  3. Dispatches tasks dynamically across reserved GPUs.
  4. Aggregates per-task JSONs into summary.json using the selected benchmark's
     official scoring protocol.

Run from the validator repo root (env managed by uv, see pyproject.toml):
    uv run python libero_eval/run_eval.py --model <path-or-hf-repo> --commit-id <hf-commit-sha> [options]
"""

import argparse
import dataclasses
import datetime
import fcntl
import heapq
import http.client
import json
import os
import pathlib
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

from backbones import BACKBONES, resolve_backbone
from axis_backend import AXIS_BENCHMARKS, apply_manifest_defaults
from axis_jax_runtime import configure_axis_jax_environment
from axis_runtime import AXIS_V1_NAME
from benchmarks import BENCHMARKS, get_benchmark
from check_model import (
    ARCHITECTURE_CONFIGS,
    MODEL_FAMILIES,
    check_lingbot_data_contract,
    check_lingbot_vla_v2_model,
    check_model_for_architectures,
    check_openvla_oft_model,
    detect_model_family,
    parse_model_architectures,
)
from download import COMMIT_HASH_RE, DEFAULT_STRATEGIES, DownloadError, download_model, parse_strategies
from download import MODEL_MAX_BYTES, ModelSizeExceeded, check_local_model_size
from gpu_health import check_gpu_availability, check_gpu_health
from lingbot_runtime import (
    DEFAULT_LINGBOT_DATA_CONTRACT,
    DEFAULT_LINGBOT_NORM_STATS,
    DEFAULT_LINGBOT_ROBOT_CONFIG_ROOT,
    LINGBOT_COMPILE_THREADS,
    LINGBOT_TORCH_INTEROP_THREADS,
    LINGBOT_TORCH_THREADS,
    runtime_contract_metadata,
)
from lingbot_eval_protocol import POLICY_BATCH_MODE_STATIC, POLICY_RNG_MODE
from paths import (
    AXIS_VENV_PY,
    CLIENT_VENV_PY,
    OPENPI_DIR,
    OPENVLA_OFT_DIR,
    OPENVLA_OFT_VENV_PY,
    ROBODOJO_DIR,
    ROBOTWIN_DIR,
    SERVER_VENV_PY,
    LINGBOT_VLA_V2_DIR,
    LINGBOT_VLA_V2_VENV_PY,
    QWEN3_VL_DIR,
    ROBOTWIN_VENV_PY,
    VALIDATOR_ROOT,
)

EVAL_TASK_SCRIPT = pathlib.Path(__file__).resolve().parent / "eval_task.py"
RELEASE_POLICY_BATCH_LANE_SCRIPT = pathlib.Path(__file__).resolve().parent / "release_policy_batch_lane.py"


def emit_progress_event(progress_file: pathlib.Path | None, detail: dict) -> None:
    """Append one complete progress event for the parent benchmark worker."""
    if progress_file is None:
        return
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with progress_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": "evaluating", "detail": detail}, sort_keys=True) + "\n")


def acquire_gpu_locks(gpus: list[int]) -> list:
    """Hold advisory per-GPU locks for the lifetime of an evaluation process."""
    if not gpus:
        raise ValueError("At least one GPU must be selected.")
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"GPU ids must be unique, got {gpus!r}.")

    lock_dir = pathlib.Path(os.environ.get("LIBERO_EVAL_GPU_LOCK_DIR", "/tmp/libero-eval-gpu-locks"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for gpu in sorted(gpus):
            handle = (lock_dir / f"gpu-{gpu}.lock").open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.seek(0)
                owner = handle.read().strip() or "unknown process"
                handle.close()
                raise RuntimeError(f"GPU {gpu} is locked by {owner}") from exc
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()} argv={' '.join(sys.argv)}\n")
            handle.flush()
            handles.append(handle)
    except Exception:
        for handle in handles:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        raise
    return handles


# ----------------------------------------------------------------------------
# Model resolution
# ----------------------------------------------------------------------------
def resolve_model(
    model: str,
    download_dir: pathlib.Path,
    strategies: list[str] | None = None,
    commit_id: str | None = None,
    repo_type: str = "model",
    subdir: str | None = None,
    ignore_patterns: list[str] | None = None,
    model_family: str = "auto",
) -> pathlib.Path:
    """Resolve a model reference (local path / HF repo id / HF URL) to a local checkpoint dir.

    HF references are downloaded pinned to `commit_id` (full 40-hex sha), and
    the download dir is keyed by the commit (same layout as benchmark_worker:
    `<user>__<repo>@<commit12>`) — different commits never share a cache dir.
    Local paths are used as-is; commit_id is only recorded in summary.json.
    """
    # Resolve to an absolute path: the policy server subprocess runs with its
    # cwd inside the openpi checkout, so a relative --model would break there.
    if repo_type not in ("model", "dataset"):
        raise ValueError(f"unsupported Hugging Face repo type {repo_type!r}")
    normalized_subdir = None
    if subdir:
        candidate = pathlib.PurePosixPath(subdir)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("--model-subdir must be a relative path without '..'")
        normalized_subdir = candidate.as_posix().strip("/")
        if not normalized_subdir or normalized_subdir == ".":
            raise ValueError("--model-subdir must name a checkpoint directory")

    # Auto-detection happens after download; use the largest supported submission
    # budget until the family-specific format check applies its tighter limit.
    max_total_bytes = max(MODEL_MAX_BYTES.values()) if model_family == "auto" else MODEL_MAX_BYTES.get(model_family)
    local = pathlib.Path(model).expanduser()
    if local.exists():
        root = local.resolve() / normalized_subdir if normalized_subdir else local.resolve()
        check_local_model_size(root, max_total_bytes)
        return _find_checkpoint_root(root)

    # Not a local path -> treat as Hugging Face reference.
    repo_id = model
    for prefix in ("https://huggingface.co/", "http://huggingface.co/", "hf://"):
        if repo_id.startswith(prefix):
            repo_id = repo_id[len(prefix) :]
            break
    repo_id = repo_id.strip("/")
    # Tolerate browser URLs pasted with a subpage suffix (user/repo/tree/main etc).
    parts = repo_id.split("/")
    if len(parts) > 2 and parts[2] in ("tree", "blob", "resolve", "commits", "settings"):
        repo_id = "/".join(parts[:2])
    if repo_id.count("/") != 1:
        raise ValueError(f"'{model}' is neither an existing local path nor a valid HF repo id (expected 'user/repo').")
    if not commit_id or not COMMIT_HASH_RE.match(commit_id):
        raise ValueError(
            f"--commit-id {commit_id!r} is not a full 40-char hex commit hash. Evaluating an HF repo "
            "requires pinning the exact commit (branch names drift); copy it from the repo's commits page."
        )

    print(f"[run_eval] Downloading HF model '{repo_id}@{commit_id[:12]}' ...")
    type_suffix = "__dataset" if repo_type == "dataset" else ""
    local_dir = (download_dir / f"{repo_id.replace('/', '__')}{type_suffix}@{commit_id[:12]}").resolve()
    allow_patterns = (
        [f"{normalized_subdir}/**", "lingbotvla_cli.yaml"]
        if normalized_subdir and repo_type == "model"
        else ([f"{normalized_subdir}/**"] if normalized_subdir else None)
    )
    download_model(
        repo_id,
        local_dir,
        revision=commit_id,
        strategies=strategies,
        repo_type=repo_type,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        max_total_bytes=max_total_bytes,
    )
    root = local_dir / normalized_subdir if normalized_subdir else local_dir
    return _find_checkpoint_root(root)


def _find_checkpoint_root(path: pathlib.Path) -> pathlib.Path:
    """Find the directory that actually contains a supported checkpoint.

    Accepts a JAX checkpoint (params/), a single-file PyTorch checkpoint, or a
    sharded Hugging Face checkpoint.  The official LingBot-VLA 2.0 release is
    nested three levels deep as `checkpoints/<step>/hf_ckpt/`.
    """

    def is_ckpt(p: pathlib.Path) -> bool:
        return (
            (p / "params").is_dir()
            or (p / "model.safetensors").is_file()
            or (p / "model.safetensors.index.json").is_file()
        )

    if is_ckpt(path):
        return path
    candidates = sorted(p.parent for p in path.glob("*/params") if p.parent != path)
    candidates += sorted(p.parent for p in path.glob("*/*/params"))
    candidates += sorted(p.parent for p in path.glob("*/model.safetensors"))
    candidates += sorted(p.parent for p in path.glob("*/*/model.safetensors"))
    candidates += sorted(p.parent for p in path.glob("*/model.safetensors.index.json"))
    candidates += sorted(p.parent for p in path.glob("*/*/model.safetensors.index.json"))
    candidates += sorted(p.parent for p in path.glob("*/*/*/params"))
    candidates += sorted(p.parent for p in path.glob("*/*/*/model.safetensors"))
    candidates += sorted(p.parent for p in path.glob("*/*/*/model.safetensors.index.json"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No supported checkpoint found under {path} (expected params/, model.safetensors, "
        "or model.safetensors.index.json, up to three nested directories)."
    )


# ----------------------------------------------------------------------------
# Policy servers
# ----------------------------------------------------------------------------
@dataclasses.dataclass
class PolicyServer:
    gpu: int
    port: int
    proc: subprocess.Popen
    log_path: pathlib.Path


def _find_free_ports(base_port: int, count: int) -> list[int]:
    """Pick `count` free TCP ports scanning upward from base_port.

    Ports occupied by unrelated services (e.g. the backend API on :8001) are
    skipped, so a foreign listener can never masquerade as one of our policy
    servers.
    """
    ports = []
    port = base_port
    while len(ports) < count:
        if port > base_port + 200:
            raise RuntimeError(f"No {count} free ports found in [{base_port}, {port}]")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                ports.append(port)
            except OSError:
                print(f"[run_eval] port {port} is in use by another service, skipping")
        port += 1
    return ports


def start_servers(
    gpus: list[int],
    base_port: int,
    config: str,
    ckpt_dir: pathlib.Path,
    log_dir: pathlib.Path,
    mem_fraction: float,
    server_impl: str = "upstream",
    max_batch: int = 4,
    model_family: str = "openpi",
    seed: int = 7,
    lingbot_norm_stats: pathlib.Path | None = None,
    lingbot_robot_config_root: pathlib.Path | None = None,
    lingbot_data_contract: pathlib.Path | None = None,
    lingbot_compile: bool = True,
    qwen3_vl_path: pathlib.Path | None = None,
    lane_count: int = 8,
    benchmark: str | None = None,
    axis_gripper_mode: str = "continuous",
    axis_policy_samples: int = 1,
    axis_sample_reduction: str = "mean",
) -> list[PolicyServer]:
    ports = _find_free_ports(base_port, len(gpus))
    is_pytorch = model_family == "openpi" and (ckpt_dir / "model.safetensors").is_file()
    axis_jax = benchmark in AXIS_BENCHMARKS and model_family == "openpi" and not is_pytorch
    # AXIS JAX must not inherit disk-cached numerical choices. Other runtimes
    # keep their existing shared compilation cache for faster startup.
    jax_cache_dir = VALIDATOR_ROOT / ".cache" / "jax_compilation"
    if not axis_jax:
        jax_cache_dir.mkdir(parents=True, exist_ok=True)
    if is_pytorch:
        print(
            "[run_eval] PyTorch checkpoint detected (model.safetensors); "
            "servers will run torch on GPU with JAX pinned to CPU"
        )
    servers = []
    for gpu, port in zip(gpus, ports):
        log_path = log_dir / f"server_gpu{gpu}.log"
        env = {
            **_base_env(),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": str(mem_fraction),
        }
        if axis_jax:
            configure_axis_jax_environment(env)
        else:
            env["JAX_COMPILATION_CACHE_DIR"] = str(jax_cache_dir)
        if qwen3_vl_path is not None:
            env["QWEN3VL_PATH"] = str(qwen3_vl_path)
        if is_pytorch:
            # serve_policy imports jax even on the PyTorch path; keep it off the
            # GPU so it cannot preallocate mem_fraction of VRAM ahead of torch.
            env["JAX_PLATFORMS"] = "cpu"
        if model_family == "lingbot_vla_v2":
            if (
                lingbot_norm_stats is None
                or lingbot_robot_config_root is None
                or lingbot_data_contract is None
                or qwen3_vl_path is None
            ):
                raise ValueError(
                    "LingBot-VLA 2.0 LIBERO serving requires norm stats, data/robot contracts and Qwen3-VL path"
                )
            # LingBot inference is GPU-bound.  Without explicit limits every
            # server creates host-wide PyTorch/OpenMP pools; seven servers then
            # starve the 21 MuJoCo clients that feed them.
            env.update({
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": str(LINGBOT_TORCH_THREADS),
                "MKL_NUM_THREADS": str(LINGBOT_TORCH_THREADS),
                "TORCHINDUCTOR_COMPILE_THREADS": str(LINGBOT_COMPILE_THREADS),
                "TORCHINDUCTOR_MAX_AUTOTUNE": "0",
                "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM": "0",
                "PYTHONHASHSEED": str(seed),
            })
            cmd = [
                str(LINGBOT_VLA_V2_VENV_PY),
                str(pathlib.Path(__file__).resolve().parent / "serve_lingbot_vla_v2.py"),
                "--port",
                str(port),
                "--model-path",
                str(ckpt_dir),
                "--robot-config-root",
                str(lingbot_robot_config_root),
                "--data-contract",
                str(lingbot_data_contract),
                "--norm-stats",
                str(lingbot_norm_stats),
                "--use-length",
                "5",
                "--use-bf16",
                "True",
                "--use-compile",
                str(bool(lingbot_compile)),
                "--seed",
                str(seed),
                "--torch-threads",
                str(LINGBOT_TORCH_THREADS),
                "--torch-interop-threads",
                str(LINGBOT_TORCH_INTEROP_THREADS),
                "--dynamic-batching",
                str(server_impl == "batched"),
                "--static-batching",
                str(server_impl == "static"),
                "--max-batch",
                str(max_batch),
                "--lane-count",
                str(lane_count),
                "--deterministic",
                str(server_impl == "static"),
            ]
            server_cwd = LINGBOT_VLA_V2_DIR
        elif model_family == "openvla_oft":
            if server_impl == "static":
                raise ValueError("static batching is currently supported only for LingBot-VLA 2.0")
            # TensorFlow is used only for the official JPEG/Lanczos + center-crop
            # preprocessing.  Its default is "all host cores per process"; with
            # one server per GPU that causes catastrophic CPU oversubscription.
            env.update({
                "TF_NUM_INTRAOP_THREADS": "2",
                "TF_NUM_INTEROP_THREADS": "1",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENVLA_TORCH_THREADS": "2",
            })
            cmd = [
                str(OPENVLA_OFT_VENV_PY),
                str(pathlib.Path(__file__).resolve().parent / "serve_openvla_oft.py"),
                "--port",
                str(port),
                "--checkpoint",
                str(ckpt_dir),
                "--seed",
                str(seed),
            ]
            server_cwd = OPENVLA_OFT_DIR
        elif benchmark in AXIS_BENCHMARKS:
            if model_family != "openpi" or server_impl != "upstream":
                raise ValueError(f"{benchmark} native OpenPI serving currently requires --server-impl upstream")
            if config != "pi05_axis_joint":
                raise ValueError(f"{benchmark} native checkpoint requires --config pi05_axis_joint")
            if is_pytorch:
                # This branch uses JAX only on CPU; its PyTorch numerical
                # behavior is not covered by the AXIS JAX runtime contract.
                env["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_exclude_nondeterministic_ops=true"
            cmd = [
                str(SERVER_VENV_PY),
                str(pathlib.Path(__file__).resolve().parent / "serve_axis_openpi.py"),
                "--port",
                str(port),
                "--config",
                config,
                "--checkpoint",
                str(ckpt_dir),
                "--openpi-root",
                str(OPENPI_DIR),
                "--gripper-mode",
                axis_gripper_mode,
                "--policy-samples",
                str(axis_policy_samples),
                "--sample-reduction",
                axis_sample_reduction,
                "--seed",
                str(seed),
            ]
            server_cwd = VALIDATOR_ROOT
        elif server_impl == "batched":
            cmd = [
                str(SERVER_VENV_PY),
                str(pathlib.Path(__file__).resolve().parent / "serve_policy_batched.py"),
                "--port",
                str(port),
                "--config",
                config,
                "--dir",
                str(ckpt_dir),
                "--max-batch",
                str(max_batch),
            ]
            server_cwd = OPENPI_DIR
        else:
            if server_impl == "static":
                raise ValueError("static batching is currently supported only for LingBot-VLA 2.0")
            cmd = [
                str(SERVER_VENV_PY),
                "scripts/serve_policy.py",
                "--port",
                str(port),
                "policy:checkpoint",
                "--policy.config",
                config,
                "--policy.dir",
                str(ckpt_dir),
            ]
            server_cwd = OPENPI_DIR
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(server_cwd),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        servers.append(PolicyServer(gpu=gpu, port=port, proc=proc, log_path=log_path))
        print(f"[run_eval] policy server: gpu={gpu} port={port} pid={proc.pid} log={log_path}")
    return servers


def _base_env() -> dict:
    import os

    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def _mujoco_egl_device_id(gpu: int) -> str:
    """Allow containers with a remapped/single EGL device to override the physical GPU id."""
    return os.environ.get("MUJOCO_EGL_DEVICE_ID") or str(gpu)


def _client_env(gpu: int, bench) -> dict:
    """Build a bounded environment for one simulator client subprocess.

    NumPy/OpenBLAS can otherwise create one host-sized thread pool per LIBERO
    client.  Static LingBot evaluation intentionally runs 56 clients, so those
    implicit pools oversubscribe the host without helping the single-threaded
    MuJoCo control loop.
    """
    return {
        **_base_env(),
        **bench.client_env(),
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": _mujoco_egl_device_id(gpu),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def _policy_server_is_healthy(port: int, timeout_s: float = 1.0) -> bool:
    """Check the HTTP health endpoint without creating a broken WebSocket handshake."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_s)
    try:
        connection.request("GET", "/healthz")
        response = connection.getresponse()
        response.read()
        return response.status == 200
    finally:
        connection.close()


def wait_for_servers(servers: list[PolicyServer], timeout_s: float) -> None:
    """Block until every server reports healthy (model loaded), or raise."""
    deadline = time.time() + timeout_s
    pending = {s.port: s for s in servers}
    while pending:
        for port, s in list(pending.items()):
            if s.proc.poll() is not None:
                tail = _tail(s.log_path)
                raise RuntimeError(
                    f"Policy server on gpu {s.gpu} (port {port}) exited with code {s.proc.returncode}.\n"
                    f"--- last log lines ({s.log_path}) ---\n{tail}"
                )
            try:
                if _policy_server_is_healthy(port) and s.proc.poll() is None:
                    print(f"[run_eval] server ready: gpu={s.gpu} port={port}")
                    del pending[port]
            except (OSError, http.client.HTTPException):
                pass
        if pending:
            if time.time() > deadline:
                raise TimeoutError(f"Servers not ready after {timeout_s}s: ports {sorted(pending)}")
            time.sleep(2)


def stop_servers(servers: list[PolicyServer]) -> None:
    for s in servers:
        if s.proc.poll() is None:
            try:
                s.proc.terminate()
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for s in servers:
        try:
            s.proc.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                s.proc.kill()
            except ProcessLookupError:
                pass
    # kill() sends a signal; it does not reap a child. Bound this second wait
    # across all servers because a driver-blocked child may never exit.
    deadline = time.monotonic() + 5
    for s in servers:
        try:
            s.proc.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            print(
                f"[run_eval] policy server pid={s.proc.pid} gpu={s.gpu} did not exit after SIGKILL; "
                "GPU driver may be blocked",
                file=sys.stderr,
            )


def _tail(path: pathlib.Path, n: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "<no log>"


# ----------------------------------------------------------------------------
# Seeded init states (anti-overfitting)
# ----------------------------------------------------------------------------
GEN_INIT_SCRIPT = pathlib.Path(__file__).resolve().parent / "gen_init_states.py"
# Cache roots per benchmark: the same (seed, num_inits) re-used across runs
# resolves to the same directory and is generated once. Production seeds come
# from the backend queue (one per miner per round), so most runs generate
# fresh; each dir is a few MB at production trial counts — prune manually if
# disk matters.
_SEEDED_INIT_CACHE = {
    "libero": pathlib.Path("~/.cache/libero_custom").expanduser(),
    "libero_pro": pathlib.Path("~/.cache/libero_pro_custom").expanduser(),
}
# Official LIBERO / LIBERO-Pro init files contain 50 states per task. Mixed
# evaluation favours the official side for odd trial counts, so only the
# remainder needs to be generated and cached.
_OFFICIAL_INITS_PER_TASK = 50


def _init_log(message):
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"{timestamp} [gen init states] {message}", flush=True)


def _count_completed_seeded_tasks(root, suites):
    """Count manifest-backed task files, tolerating an in-progress manifest write."""
    completed = 0
    for suite in suites:
        manifest_path = root / suite / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        tasks = manifest.get("tasks", {})
        if isinstance(tasks, dict):
            completed += len(tasks)
    return completed


def _required_seeded_inits(num_trials, official_inits=_OFFICIAL_INITS_PER_TASK):
    n_official = min((num_trials + 1) // 2, official_inits)
    return max(0, num_trials - n_official)


def _init_task_shards(suites, suite_n_tasks, gpus, workers_per_gpu):
    if workers_per_gpu < 1:
        raise ValueError("init workers per GPU must be at least 1")
    slots = [(gpu, worker_id) for gpu in gpus for worker_id in range(workers_per_gpu)]
    shards = {slot: [] for slot in slots}
    tasks = [(suite, task_id) for suite in suites for task_id in range(suite_n_tasks[suite])]
    for i, task in enumerate(tasks):
        shards[slots[i % len(slots)]].append(task)
    return [(gpu, worker_id, tasks) for (gpu, worker_id), tasks in shards.items() if tasks]


def ensure_seeded_init_states(seed, num_inits, suites, suite_n_tasks, gpus, init_workers_per_gpu, bench, log_dir):
    """Generate the seeded init states for `suites` (idempotent) and return the root.

    Individual (suite, task_id) pairs are round-robined across GPU worker slots
    (runs before the policy servers claim the cards). The script skips task
    files that already exist, so a warm cache costs only process startup.

    num_inits is part of the cache dir name: gen_init_states refuses to mix
    counts within one root (manifest guard). Generation is prefix-stable (same
    seed, the first k states are identical for any num_inits >= k), so a
    smaller count is a true subset of a larger one — dirs never contradict
    each other, they just don't share files.
    """
    root = _SEEDED_INIT_CACHE[bench.name] / f"init_files_seed{seed}_n{num_inits}"
    total_tasks = sum(suite_n_tasks[suite] for suite in suites)
    shards = _init_task_shards(suites, suite_n_tasks, gpus, init_workers_per_gpu)

    procs = []
    for gpu, worker_id, tasks in shards:
        cmd = [
            str(CLIENT_VENV_PY),
            str(GEN_INIT_SCRIPT),
            "--seed",
            str(seed),
            "--num-inits",
            str(num_inits),
            "--task-specs",
            ",".join(f"{suite}:{task_id}" for suite, task_id in tasks),
            "--output-root",
            str(root),
        ]
        env = {
            **_base_env(),
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": _mujoco_egl_device_id(gpu),
            **bench.client_env(),
        }
        log_path = log_dir / f"gen_init_seed{seed}_gpu{gpu}_worker{worker_id}.log"
        log_f = open(log_path, "w")
        procs.append((gpu, worker_id, subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT), log_f))

    started = time.monotonic()
    last_completed = -1
    last_report = 0.0

    def report_progress(force=False):
        nonlocal last_completed, last_report
        now = time.monotonic()
        completed = min(_count_completed_seeded_tasks(root, suites), total_tasks)
        if not force and completed == last_completed and now - last_report < 30:
            return
        active_workers = sum(proc.poll() is None for _, _, proc, _ in procs)
        _init_log(
            f"init-state progress: seed={seed} tasks={completed}/{total_tasks} "
            f"remaining={total_tasks - completed} active_workers={active_workers} "
            f"elapsed={now - started:.0f}s"
        )
        last_completed = completed
        last_report = now

    report_progress(force=True)
    while any(proc.poll() is None for _, _, proc, _ in procs):
        time.sleep(1)
        report_progress()
    report_progress(force=True)

    failed = []
    for gpu, worker_id, proc, log_f in procs:
        ret = proc.wait()
        log_f.close()
        if ret != 0:
            failed.append(f"gpu{gpu}/worker{worker_id}")
    if failed:
        sys.exit(
            f"[run_eval] seeded init-state generation failed on {failed}; "
            f"see {log_dir}/gen_init_seed{seed}_gpu*_worker*.log"
        )
    return root


# ----------------------------------------------------------------------------
# Task dispatch
# ----------------------------------------------------------------------------
@dataclasses.dataclass
class TaskSpec:
    suite: str
    task_id: int
    max_steps: int
    scheduling_cost: float | None = None

    @property
    def name(self) -> str:
        return f"{self.suite}_task{self.task_id:02d}"

    @property
    def dispatch_cost(self) -> float:
        return float(self.max_steps if self.scheduling_cost is None else self.scheduling_cost)


# Immutable scheduling estimates from the median per-suite task runtime of two
# complete LingBot deterministic/static production baselines. ``max_steps``
# badly underestimates failure-heavy spatial swap tasks and used to place them
# behind the longest 520-step tasks, creating a 30-45 minute tail. These values
# affect only the fixed task-to-lane plan, never model inputs, outputs, or score.
# Keep the version stable so evaluator_source_git_commit fully identifies the
# schedule used for a submitted result.
LINGBOT_LIBERO_PRO_STATIC_SCHEDULING_PROFILE = "lingbot_libero_pro_suite_runtime_v1"
LINGBOT_LIBERO_PRO_STATIC_SUITE_COSTS = {
    "libero_spatial_object": 1589.0,
    "libero_spatial_swap": 2293.0,
    "libero_spatial_lan": 1557.0,
    "libero_spatial_task": 2254.0,
    "libero_object_object": 2537.0,
    "libero_object_swap": 3404.0,
    "libero_object_lan": 2384.0,
    "libero_object_task": 3615.0,
    "libero_goal_object": 2235.0,
    "libero_goal_swap": 3510.0,
    "libero_goal_lan": 1877.0,
    "libero_goal_task": 3538.0,
    "libero_10_object": 4514.0,
    "libero_10_swap": 5844.0,
    "libero_10_lan": 3613.0,
    "libero_10_task": 5883.0,
}


def static_scheduling_profile(benchmark_name: str) -> str:
    if benchmark_name == "libero_pro":
        return LINGBOT_LIBERO_PRO_STATIC_SCHEDULING_PROFILE
    return "max_steps_v1"


def static_scheduling_cost(benchmark_name: str, suite: str, max_steps: int) -> float:
    if benchmark_name == "libero_pro":
        try:
            return LINGBOT_LIBERO_PRO_STATIC_SUITE_COSTS[suite]
        except KeyError as exc:
            raise ValueError(f"no fixed LingBot static scheduling cost for LIBERO-Pro suite {suite!r}") from exc
    return float(max_steps)


def partition_static_lanes(specs: list[TaskSpec], lane_count: int) -> list[list[TaskSpec]]:
    """Deterministically balance an already highest-cost-first task list over fixed lanes."""
    if lane_count < 1:
        raise ValueError(f"lane_count must be positive, got {lane_count}")
    lanes: list[list[TaskSpec]] = [[] for _ in range(lane_count)]
    heap = [(0, lane) for lane in range(lane_count)]
    heapq.heapify(heap)
    for spec in specs:
        load, lane = heapq.heappop(heap)
        lanes[lane].append(spec)
        heapq.heappush(heap, (load + spec.dispatch_cost, lane))
    return lanes


def interleaved_static_lane_index(
    server_index: int,
    lane_id: int,
    server_count: int,
    cohort_size: int,
) -> int:
    """Map a GPU-local lane to the global LPT plan by whole fixed cohorts.

    LPT gives neighboring plan lanes similar assignments. Mapping contiguous
    blocks of eight to GPUs can therefore put every short two-task lane on one
    GPU and leave that card idle during the tail. Interleave complete cohorts
    across GPUs, but never transpose individual lanes: keeping adjacent plan
    lanes together preserves prompt-length locality inside each exact-shape
    model batch and avoids unnecessary sequence padding.
    """
    if server_count < 1:
        raise ValueError(f"server_count must be positive, got {server_count}")
    if cohort_size < 1:
        raise ValueError(f"cohort_size must be positive, got {cohort_size}")
    if not 0 <= server_index < server_count:
        raise ValueError(f"server_index {server_index} is outside server_count={server_count}")
    if lane_id < 0:
        raise ValueError(f"lane_id must be non-negative, got {lane_id}")
    cohort_id, slot_id = divmod(lane_id, cohort_size)
    return (cohort_id * server_count + server_index) * cohort_size + slot_id


def split_resumable(specs: list, out_dir: pathlib.Path) -> tuple[list, dict]:
    """--resume: split specs into (still to run, results preloaded from disk).

    A task counts as done when its result JSON exists and parses (a process
    killed mid-write leaves an unparseable file -> rerun). Failed tasks never
    write a result JSON, so they rerun too.
    """
    todo, results = [], {}
    for spec in specs:
        path = out_dir / "results" / f"{spec.name}.json"
        try:
            task_result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            todo.append(spec)
            continue
        results[spec.name] = {"status": "ok", "gpu": None, "attempts": 0, "resumed": True, **task_result}
    return todo, results


def release_static_lane(server: PolicyServer, lane_id: int) -> None:
    """Tell the server that this lane has no more tasks.

    eval_task connections intentionally come and go between tasks, so socket
    closure cannot carry this meaning. Keep the control request in the same
    client environment and wire codec as scored requests.
    """
    cmd = [
        str(CLIENT_VENV_PY),
        str(RELEASE_POLICY_BATCH_LANE_SCRIPT),
        "--host",
        "127.0.0.1",
        "--port",
        str(server.port),
        "--lane",
        str(lane_id),
    ]
    try:
        completed = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[run_eval] WARNING: failed to release static lane {lane_id} on gpu={server.gpu}: {exc}")
        return
    if completed.returncode:
        detail = completed.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        print(
            f"[run_eval] WARNING: static lane {lane_id} release failed on gpu={server.gpu} "
            f"with exit {completed.returncode}{suffix}"
        )


def worker_loop(
    server: PolicyServer,
    task_q: "queue.Queue[TaskSpec]",
    args,
    bench,
    out_dir: pathlib.Path,
    results: dict,
    lock: threading.Lock,
    progress: dict,
    lane_id: int | None = None,
) -> None:
    while True:
        try:
            spec = task_q.get_nowait()
        except queue.Empty:
            if args.server_impl == "static" and lane_id is not None:
                release_static_lane(server, lane_id)
            return

        result_json = out_dir / "results" / f"{spec.name}.json"
        log_path = out_dir / "logs" / f"{spec.name}.log"
        cmd = [
            str(CLIENT_VENV_PY),
            str(EVAL_TASK_SCRIPT),
            "--host",
            "127.0.0.1",
            "--port",
            str(server.port),
            "--task-suite-name",
            spec.suite,
            "--task-id",
            str(spec.task_id),
            "--max-steps",
            str(spec.max_steps),
            "--prompt-source",
            bench.prompt_source,
            "--num-trials",
            str(args.num_trials),
            "--seed",
            str(args.seed),
            "--out-json",
            str(result_json),
        ]
        if args.model_family == "openvla_oft":
            cmd += ["--policy-preprocess", "openvla_oft", "--resize-size", "256", "--replan-steps", "8"]
        elif args.model_family == "lingbot_vla_v2":
            cmd += ["--policy-preprocess", "lingbot_vla_v2", "--resize-size", "256", "--replan-steps", "5"]
            if args.server_impl == "static" and lane_id is not None:
                cmd += ["--policy-batch-lane", str(lane_id)]
        if args.init_states_root:
            cmd += ["--init-states-root", str(args.init_states_root)]
            if args.init_states_mix:
                cmd += ["--init-states-mix"]
        if args.save_videos > 0:
            cmd += ["--video-dir", str(out_dir / "videos"), "--save-videos", str(args.save_videos)]

        env = _client_env(server.gpu, bench)

        # Generous per-task timeout so a hung simulator can't stall the whole run.
        task_timeout = args.task_timeout or (args.num_trials * 200 + 900)
        status, attempts = "failed", 0
        t0 = time.time()
        for attempt in range(args.retries + 1):
            attempts = attempt + 1
            with open(log_path, "a") as log_f:
                log_f.write(f"\n===== attempt {attempts} (gpu {server.gpu}, port {server.port}) =====\n")
                log_f.flush()
                try:
                    ret = subprocess.run(
                        cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, timeout=task_timeout
                    ).returncode
                except subprocess.TimeoutExpired:
                    log_f.write(f"\n===== TIMEOUT after {task_timeout}s =====\n")
                    ret = -1
            if ret == 0 and result_json.exists():
                status = "ok"
                break

        with lock:
            progress["done"] += 1
            done, total = progress["done"], progress["total"]
            progress["suite_done"][spec.suite] += 1
            if status == "ok":
                task_result = json.loads(result_json.read_text())
                results[spec.name] = {"status": "ok", "gpu": server.gpu, "attempts": attempts, **task_result}
                progress["episodes_done"] += task_result.get("num_trials", 0)
                print(
                    f"[run_eval] [{done}/{total}] {spec.name}: "
                    f"{task_result['num_successes']}/{task_result['num_trials']} "
                    f"({task_result['success_rate']:.0%}) gpu={server.gpu} {time.time() - t0:.0f}s"
                )
            else:
                results[spec.name] = {
                    "status": "failed",
                    "gpu": server.gpu,
                    "attempts": attempts,
                    "task_suite_name": spec.suite,
                    "task_id": spec.task_id,
                }
                print(f"[run_eval] [{done}/{total}] {spec.name}: FAILED (see {log_path})")

            if progress["suite_done"][spec.suite] == progress["suite_total"][spec.suite]:
                progress["suites_done"] += 1
                emit_progress_event(
                    progress["progress_file"],
                    {
                        "suites_done": progress["suites_done"],
                        "suites_total": progress["suites_total"],
                        "last_completed_suite": spec.suite,
                        "episodes_done": progress["episodes_done"],
                        "episodes_total": progress["episodes_total"],
                    },
                )


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def _rollup(results: dict, label_fn) -> dict:
    """Aggregate per-task results into {label: {tasks, failed_tasks, episodes, successes, success_rate}}.

    label_fn maps a task result to its group label; None-labeled results are
    skipped (e.g. tasks outside the group mapping).
    """
    groups: dict = {}
    for _, r in sorted(results.items()):
        label = label_fn(r)
        if label is None:
            continue
        agg = groups.setdefault(label, {"tasks": 0, "failed_tasks": 0, "episodes": 0, "successes": 0})
        agg["tasks"] += 1
        if r["status"] != "ok":
            agg["failed_tasks"] += 1
            continue
        agg["episodes"] += r["num_trials"]
        agg["successes"] += r["num_successes"]
    for agg in groups.values():
        agg["success_rate"] = agg["successes"] / agg["episodes"] if agg["episodes"] else None
    return groups


def summarize(results: dict, meta: dict, task_groups: dict | None = None) -> dict:
    """Aggregate results per suite (and, with task_groups = {suite: {task_id:
    label}}, per group — LIBERO-plus's perturbation dimensions)."""
    suites = _rollup(results, lambda r: r["task_suite_name"])
    total_eps = sum(a["episodes"] for a in suites.values())
    total_succ = sum(a["successes"] for a in suites.values())
    summary = {
        **meta,
        "suites": suites,
        "total_episodes": total_eps,
        "total_successes": total_succ,
        "total_success_rate": total_succ / total_eps if total_eps else None,
        "tasks": dict(sorted(results.items())),
    }
    if task_groups:
        summary["dimensions"] = _rollup(
            results, lambda r: (task_groups.get(r["task_suite_name"]) or {}).get(r["task_id"])
        )
    return summary


def print_report(summary: dict) -> None:
    dimensions = summary.get("dimensions") or {}
    name_w = max([18] + [len(n) + 1 for n in (*summary["suites"], *dimensions)])
    line_w = name_w + 6 + 10 + 10 + 9

    def print_rows(header, groups):
        print(f"{header:<{name_w}}{'tasks':>6}{'episodes':>10}{'successes':>10}{'rate':>9}")
        for name, agg in groups.items():
            rate = f"{agg['success_rate']:.1%}" if agg["success_rate"] is not None else "n/a"
            extra = f"  ({agg['failed_tasks']} failed!)" if agg["failed_tasks"] else ""
            print(f"{name:<{name_w}}{agg['tasks']:>6}{agg['episodes']:>10}{agg['successes']:>10}{rate:>9}{extra}")

    print("\n" + "=" * line_w)
    print(f"Model:    {summary['model']}")
    print(f"Benchmark: {summary.get('benchmark', 'libero')}")
    protocol = summary.get("evaluation_protocol")
    if protocol:
        status = "OFFICIAL" if protocol.get("official_result") else "NON-OFFICIAL"
        print(f"Protocol: {protocol['name']} [{status}]")
    print(f"Config:   {summary['config']}   trials/task: {summary['num_trials_per_task']}")
    print(f"GPUs:     {summary['gpus']}   wall time: {summary['wall_time_s']:.0f}s")
    print("-" * line_w)
    print_rows("suite", summary["suites"])
    if dimensions:
        print("-" * line_w)
        print_rows("dimension", dict(sorted(dimensions.items())))
    print("-" * line_w)
    rate = summary["total_success_rate"]
    print(
        f"{'TOTAL':<{name_w}}{'':>6}{summary['total_episodes']:>10}{summary['total_successes']:>10}"
        f"{(f'{rate:.1%}' if rate is not None else 'n/a'):>9}"
    )
    print("=" * line_w)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def _reject_model(message: str) -> None:
    """Exit with the worker's stable code for a permanent model rejection."""
    print(f"[run_eval] model REJECTED: {message}")
    raise SystemExit(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Local checkpoint dir, HF repo id (user/repo), or HF URL")
    parser.add_argument(
        "--model-repo-type",
        choices=("model", "dataset"),
        default="model",
        help="Hugging Face repo type (RoboDojo publishes checkpoints in a dataset repo)",
    )
    parser.add_argument(
        "--model-subdir",
        default=None,
        help="Only download/resolve this checkpoint subdirectory inside the Hugging Face repo",
    )
    parser.add_argument(
        "--backbone",
        default=None,
        metavar="NAME",
        help=(
            "Policy backbone (default: detect checkpoint family; openpi falls back to pi0.5; "
            f"choose from {', '.join(BACKBONES)})"
        ),
    )
    parser.add_argument(
        "--model-family",
        default="auto",
        choices=("auto", *MODEL_FAMILIES),
        help="Policy runtime (default: infer from checkpoint markers)",
    )
    parser.add_argument(
        "--commit-id",
        required=True,
        help="HF commit hash being evaluated (full 40-hex sha). HF models are downloaded pinned "
        "to this commit and cached per commit, so a re-submission to the same repo is always "
        "re-fetched. For a local checkpoint dir it is only recorded in summary.json "
        "(pass 'local' for ad-hoc local runs).",
    )
    parser.add_argument(
        "--evaluator-source-git-commit",
        default=None,
        help="Full Git revision of a separately deployed evaluator source bundle; recorded in summary.json",
    )
    benchmark_group = parser.add_mutually_exclusive_group()
    benchmark_group.add_argument(
        "--benchmark",
        default="libero",
        choices=sorted((*BENCHMARKS, *AXIS_BENCHMARKS, "robodojo", "robotwin")),
        metavar="BENCHMARK",
        help="Which benchmark to evaluate on; use axis_v2.0 for current AXIS (default: libero)",
    )
    benchmark_group.add_argument(
        "--axis_v1.0",
        dest="benchmark",
        action="store_const",
        const=AXIS_V1_NAME,
        help="Shortcut for --benchmark=axis_v1.0 (30 tasks configured in axis_v1.0.yaml)",
    )
    parser.add_argument(
        "--model-architectures",
        default=None,
        metavar="ARCH[,ARCH...]",
        help="Deprecated openpi architecture allow-list; use --backbone for new commands",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Explicit openpi training config name (default: auto-select from the checkpoint architecture)",
    )
    parser.add_argument(
        "--suites", default=None, help="Comma-separated suites (default: the benchmark's standard suites)"
    )
    parser.add_argument(
        "--task-ids", default=None, help="Comma-separated task ids to evaluate (default: all tasks in each suite)"
    )
    parser.add_argument(
        "--axis-sample-size",
        type=int,
        default=None,
        help="AXIS only: randomly draw N tasks from the configured pool (default: evaluate the full pool)",
    )
    parser.add_argument(
        "--axis-sampling-seed",
        type=int,
        default=None,
        help="AXIS task-selection seed; omitted means fresh local randomness, recorded in task_selection.json",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="RoboDojo/RoboTwin only: comma-separated task names",
    )
    parser.add_argument(
        "--eval-seeds",
        default="0,1,2",
        help="RoboDojo only: layout seeds (default: official 0,1,2)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Trials per task (default: benchmark protocol; RoboDojo uses native 50, or 25+25 "
        "for Generalization; a RoboDojo override applies to each simulator half)",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=None,
        help="Concurrent eval clients per GPU sharing one policy server (default: 3 for LIBERO, "
        "1 for RoboTwin). LIBERO's default 3 is measured "
        "optimum on 4090, ~1.8x over 1 — the server handles requests serially, so >1 overlaps "
        "one client's CPU simulation with another's GPU inference). Note: with >1 the JAX rng "
        "consumption order depends on request arrival order, so per-call action noise differs "
        "across runs (same situation as re-running with a different task-to-GPU layout); "
        "per-trial init states and seeds are unchanged. Pass 1 to reproduce the strictly "
        "serial legacy behavior.",
    )
    parser.add_argument(
        "--init-workers-per-gpu",
        type=int,
        default=4,
        help="Concurrent init-state generator processes per GPU (default 4; independent of eval workers)",
    )
    parser.add_argument(
        "--server-impl",
        choices=("upstream", "batched", "static"),
        default="upstream",
        help="Policy server implementation: 'upstream' = batch 1; 'batched' = greedy dynamic batching; "
        "'static' = LingBot-only fixed cohorts with strict deterministic CUDA/PyTorch settings",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=4,
        help="Upper bound on the dynamic batch size (batched server only); padded to powers of 2",
    )
    parser.add_argument("--base-port", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=None, help="Policy seed (default: benchmark config, otherwise 7)")
    parser.add_argument(
        "--init-states-root",
        default=None,
        help="Evaluate on privately re-sampled init states: load "
        "{root}/{suite}/{task}.pruned_init (generated by gen_init_states.py) instead of "
        "the benchmark's official frozen files (anti-overfitting; replaces ALL trials — "
        "for the production 50/50 mix use --init-seed)",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=None,
        help="Anti-overfitting production mode: generate (idempotently, cached per seed) "
        "privately re-sampled init states for the requested suites, then run half of each "
        "task's trials on the official init states and half on the seeded ones. "
        "Mutually exclusive with --init-states-root.",
    )
    parser.add_argument("--save-videos", type=int, default=1, help="Save videos for first N trials per task")
    parser.add_argument("--retries", type=int, default=1, help="Retries per task on failure")
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=0,
        help="Seconds before a task subprocess is killed (0 = auto: trials*200+900)",
    )
    parser.add_argument("--server-timeout", type=float, default=900, help="Seconds to wait for servers")
    parser.add_argument("--mem-fraction", type=float, default=0.7, help="XLA GPU memory fraction per server")
    parser.add_argument("--output-dir", default=None, help="Default: validator/eval_runs/<timestamp>_<model-name>")
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Append JSONL evaluation progress events for benchmark_worker (optional)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks whose result JSON already exists (crash recovery for long runs; "
        "point --output-dir at the interrupted run's directory)",
    )
    parser.add_argument(
        "--download-dir", default=str(VALIDATOR_ROOT / "hf_models"), help="Where HF models are downloaded"
    )
    parser.add_argument(
        "--download-strategies",
        default=os.environ.get("MODEL_DOWNLOAD_STRATEGIES", DEFAULT_STRATEGIES),
        help="模型下载策略顺序(逗号分隔):hfd-mirror,hfd,hub-mirror,hub",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip the pre-evaluation checkpoint format check (debugging only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="AXIS/RoboDojo/RoboTwin: validate the environment without starting a policy server",
    )
    parser.add_argument(
        "--axis-task-api-base-url",
        default=None,
        help="Override the Axis task API (base-only protocols use frozen repository snapshots)",
    )
    parser.add_argument(
        "--axis-asset-base-url",
        default=None,
        help="Override the Axis MuJoCo asset CDN from the selected frozen manifest",
    )
    parser.add_argument(
        "--axis-cache-root",
        default=str(VALIDATOR_ROOT / ".cache" / "axis"),
        help="Task-payload and scene-asset cache for Axis benchmarks",
    )
    parser.add_argument(
        "--axis-manifest",
        default=None,
        help="Explicit versioned task manifest for --benchmark axis; named Axis releases remain frozen",
    )
    parser.add_argument(
        "--axis-randomization-manifest",
        default=None,
        help="Frozen scene-variant manifest override for randomized AXIS protocols",
    )
    parser.add_argument(
        "--axis-randomization-seed",
        type=int,
        default=None,
        help="Public seed for deterministic AXIS variant selection (default: 0 for randomized protocols)",
    )
    parser.add_argument(
        "--axis-asset-fetch-workers",
        type=int,
        default=16,
        help="Concurrent first-run AXIS asset downloads (default: 16)",
    )
    parser.add_argument(
        "--axis-record-trials",
        type=int,
        default=0,
        help="Save a rollout GIF and per-step state/checker for the first N trials of every Axis task",
    )
    parser.add_argument(
        "--axis-replan-steps",
        type=int,
        default=None,
        help="Joint-target steps consumed from each AXIS policy action chunk",
    )
    parser.add_argument(
        "--axis-gripper-mode",
        choices=("continuous", "symmetric-binary"),
        default="continuous",
        help="Uniform policy output decoding, frozen by the AXIS manifest protocol",
    )
    parser.add_argument(
        "--axis-policy-samples",
        type=int,
        choices=range(1, 17),
        default=1,
        help="Independent diffusion predictions per AXIS policy request; frozen in the manifest",
    )
    parser.add_argument("--axis-sample-reduction", choices=("mean", "medoid"), default="mean")
    parser.add_argument(
        "--axis-max-control-steps",
        type=int,
        default=None,
        help="Development override for the selected Axis protocol's episode limit",
    )
    parser.add_argument(
        "--axis-policy-host",
        default="127.0.0.1",
        help="Host for an external AXIS-compatible policy server",
    )
    parser.add_argument(
        "--axis-policy-port",
        type=int,
        default=None,
        help="Use an already-running AXIS-compatible policy server instead of starting the model",
    )
    parser.add_argument(
        "--robotwin-task-config",
        choices=("demo_clean", "demo_randomized"),
        default="demo_clean",
        help="RoboTwin domain setting (default: demo_clean)",
    )
    parser.add_argument(
        "--robotwin-instruction-type",
        choices=("seen", "unseen"),
        default=None,
        help="Override RoboTwin language split for diagnostics (default: upstream deploy config, unseen)",
    )
    parser.add_argument(
        "--qwen3-vl-path",
        default=str(QWEN3_VL_DIR),
        help="Local pinned Qwen3-VL config/processor snapshot used by LingBot-VLA 2.0",
    )
    parser.add_argument(
        "--lingbot-norm-stats",
        default=str(DEFAULT_LINGBOT_NORM_STATS),
        help="Evaluator-owned normalization JSON for LingBot-VLA 2.0 on LIBERO/LIBERO-Pro",
    )
    parser.add_argument(
        "--lingbot-data-contract",
        default=str(DEFAULT_LINGBOT_DATA_CONTRACT),
        help="Evaluator-owned LingBot-VLA 2.0 LIBERO camera/state/action contract",
    )
    parser.add_argument(
        "--lingbot-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.compile in the LingBot-VLA 2.0 server (default: enabled)",
    )
    args = parser.parse_args()
    sampling_requested = args.axis_sample_size is not None or args.axis_sampling_seed is not None
    if sampling_requested and args.benchmark not in AXIS_BENCHMARKS:
        parser.error("--axis-sample-size and --axis-sampling-seed require an AXIS benchmark")
    if args.axis_manifest is not None and args.benchmark != "axis":
        parser.error("--axis-manifest requires --benchmark axis; named releases cannot be overridden")
    if args.benchmark == "axis" and args.axis_manifest is None:
        parser.error("--benchmark axis requires --axis-manifest")
    if args.benchmark in AXIS_BENCHMARKS:
        try:
            apply_manifest_defaults(args)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    if args.seed is None:
        args.seed = 7
    if args.axis_replan_steps is None:
        args.axis_replan_steps = 5
    if args.evaluator_source_git_commit is not None and not COMMIT_HASH_RE.fullmatch(args.evaluator_source_git_commit):
        parser.error("--evaluator-source-git-commit must be a full 40-character lowercase Git hash")
    if args.workers_per_gpu is None:
        args.workers_per_gpu = 1 if args.benchmark in ("robotwin", *AXIS_BENCHMARKS) else 3
    backbone_explicit = args.backbone is not None or args.model_architectures is not None or args.model_family != "auto"

    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be at least 1")
    if args.init_workers_per_gpu < 1:
        parser.error("--init-workers-per-gpu must be at least 1")

    try:
        legacy_architectures = (
            parse_model_architectures(args.model_architectures) if args.model_architectures is not None else None
        )
        backbone, model_architectures = resolve_backbone(
            args.backbone,
            legacy_architectures,
            args.model_family,
        )
    except ValueError as e:
        parser.error(str(e))
    args.backbone = backbone.name

    if args.benchmark in AXIS_BENCHMARKS:
        if args.backbone != "pi0.5":
            parser.error(f"{args.benchmark} currently requires --backbone pi0.5; no LingBot AXIS adapter is installed")
        if not AXIS_VENV_PY.exists():
            sys.exit(f"[run_eval] Missing {AXIS_VENV_PY} — install it with bash setup_axis.sh")
    elif args.benchmark not in ("robodojo", "robotwin"):
        for py, what in [(CLIENT_VENV_PY, "LIBERO client venv")]:
            if not py.exists():
                sys.exit(f"[run_eval] Missing {py} — install the {what} first (bash setup.sh; see README).")

    gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    if args.benchmark in (*AXIS_BENCHMARKS, "robodojo", "robotwin") and args.dry_run:
        if not gpus or len(gpus) != len(set(gpus)):
            parser.error("--gpus must contain at least one unique GPU id")
    else:
        # Preflight: a hung NVIDIA driver makes every CUDA/EGL process enter an
        # unkillable D state. Detect it before spawning policy/simulator processes.
        print("[run_eval] GPU preflight: checking NVIDIA driver (15s timeout)", flush=True)
        health = check_gpu_health()
        if not health.healthy:
            sys.exit(
                f"[run_eval] GPU preflight failed: {health.detail}\n"
                "The NVIDIA driver may be hung (D-state processes). A reboot or driver reload is "
                "required before evaluation can run."
            )
        try:
            # Keep these file objects alive until main exits. The OS releases the
            # flock automatically on normal, exceptional, or signalled exit.
            _gpu_locks = acquire_gpu_locks(gpus)
        except (OSError, ValueError, RuntimeError) as e:
            sys.exit(f"[run_eval] GPU reservation failed: {e}")
        # LIBERO starts its own model and renderers. External-policy backends
        # deliberately share with an existing server and keep their own policy.
        if args.benchmark in ("libero", "libero_pro", "libero_plus"):
            availability = check_gpu_availability(gpus)
            if not availability.healthy:
                sys.exit(f"[run_eval] GPU reservation failed: {availability.detail}")

    if args.benchmark in AXIS_BENCHMARKS and (args.dry_run or args.axis_policy_port is not None):
        try:
            from axis_backend import run as run_axis

            return_code = run_axis(args, None, gpus, AXIS_VENV_PY)
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            sys.exit(f"[run_eval] AXIS environment rejected: {e}")
        if return_code:
            sys.exit(return_code)
        return

    if args.benchmark == "robotwin":
        if args.init_seed is not None or args.init_states_root:
            sys.exit(
                "[run_eval] RoboTwin task layouts are selected by --robotwin-task-config; "
                "LIBERO init options do not apply"
            )
        if args.backbone != "lingbot-vla-v2":
            sys.exit(
                f"[run_eval] RoboTwin integration currently requires --backbone lingbot-vla-v2, got {args.backbone!r}"
            )
        try:
            strategies = parse_strategies(args.download_strategies)
            checkpoint = resolve_model(
                args.model,
                pathlib.Path(args.download_dir),
                strategies,
                commit_id=args.commit_id,
                repo_type=args.model_repo_type,
                subdir=args.model_subdir,
                model_family=backbone.model_family,
            )
            from robotwin_backend import run as run_robotwin

            return_code = run_robotwin(
                args,
                checkpoint,
                gpus,
                LINGBOT_VLA_V2_DIR,
                ROBOTWIN_DIR,
                LINGBOT_VLA_V2_VENV_PY,
                ROBOTWIN_VENV_PY,
                pathlib.Path(args.qwen3_vl_path).expanduser().resolve(),
            )
        except ModelSizeExceeded as e:
            _reject_model(str(e))
        except DownloadError as e:
            sys.exit(f"[run_eval] model download failed: {e}")
        except (FileNotFoundError, RuntimeError, TimeoutError, ValueError) as e:
            sys.exit(f"[run_eval] RoboTwin evaluation rejected: {e}")
        if return_code:
            sys.exit(return_code)
        return

    if args.benchmark == "robodojo":
        if args.init_seed is not None or args.init_states_root:
            sys.exit(
                "[run_eval] RoboDojo layout randomization uses --eval-seeds; LIBERO init-state options do not apply"
            )
        if args.model_family not in ("auto", "openpi"):
            sys.exit("[run_eval] RoboDojo currently supports the official Pi 0.5/Pi 0 OpenPI adapters only")
        try:
            strategies = parse_strategies(args.download_strategies)
            checkpoint_input = None
            if args.model_subdir and ("{" in args.model_subdir or "}" in args.model_subdir):
                if (
                    args.model_subdir.count("{seed}") != 1
                    or "{" in args.model_subdir.replace("{seed}", "")
                    or "}" in args.model_subdir.replace("{seed}", "")
                ):
                    raise ValueError("RoboDojo --model-subdir only supports the {seed} placeholder")
                try:
                    eval_seeds = tuple(dict.fromkeys(int(item) for item in args.eval_seeds.split(",") if item != ""))
                except ValueError as exc:
                    raise ValueError("--eval-seeds must be comma-separated integers") from exc
                checkpoint_input = {
                    seed: resolve_model(
                        args.model,
                        pathlib.Path(args.download_dir),
                        strategies,
                        commit_id=args.commit_id,
                        repo_type=args.model_repo_type,
                        subdir=args.model_subdir.replace("{seed}", str(seed)),
                        ignore_patterns=[f"{args.model_subdir.replace('{seed}', str(seed))}/train_state/**"],
                        model_family=backbone.model_family,
                    )
                    for seed in eval_seeds
                }
            else:
                checkpoint_input = resolve_model(
                    args.model,
                    pathlib.Path(args.download_dir),
                    strategies,
                    commit_id=args.commit_id,
                    repo_type=args.model_repo_type,
                    subdir=args.model_subdir,
                    ignore_patterns=[f"{args.model_subdir}/train_state/**"] if args.model_subdir else None,
                    model_family=backbone.model_family,
                )
            from robodojo_backend import run as run_robodojo

            return_code = run_robodojo(args, checkpoint_input, model_architectures, gpus, ROBODOJO_DIR)
        except ModelSizeExceeded as e:
            _reject_model(str(e))
        except DownloadError as e:
            sys.exit(f"[run_eval] model download failed: {e}")
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            sys.exit(f"[run_eval] RoboDojo evaluation rejected: {e}")
        if return_code:
            sys.exit(return_code)
        return

    is_axis = args.benchmark in AXIS_BENCHMARKS
    bench = None if is_axis else get_benchmark(args.benchmark)
    suites = (
        []
        if is_axis
        else ([s.strip() for s in args.suites.split(",") if s.strip()] if args.suites else list(bench.default_suites))
    )
    task_ids = [int(t) for t in args.task_ids.split(",")] if args.task_ids else None
    if args.num_trials is None and not is_axis:
        args.num_trials = bench.default_num_trials

    args.init_states_mix = False
    if args.init_states_root and args.init_seed is not None:
        sys.exit("[run_eval] --init-seed and --init-states-root are mutually exclusive")
    if not is_axis:
        if (args.init_seed is not None or args.init_states_root) and not bench.supports_seeded_inits:
            sys.exit(
                f"[run_eval] --init-seed/--init-states-root are not supported for benchmark '{bench.name}': "
                "its task variants carry their own perturbation-specific init states that must not be resampled"
            )
        if args.init_states_root:
            args.init_states_root = str(pathlib.Path(args.init_states_root).expanduser().resolve())
            missing = [s for s in suites if not (pathlib.Path(args.init_states_root) / s).is_dir()]
            if missing:
                sys.exit(
                    f"[run_eval] --init-states-root {args.init_states_root} has no init states for "
                    f"suites {missing}; generate them first (libero_eval/gen_init_states.py)."
                )
            print(f"[run_eval] init states override: {args.init_states_root}")

    try:
        strategies = parse_strategies(args.download_strategies)
    except ValueError as e:
        sys.exit(f"[run_eval] {e}")

    try:
        ckpt_dir = resolve_model(
            args.model,
            pathlib.Path(args.download_dir),
            strategies,
            commit_id=args.commit_id,
            repo_type=args.model_repo_type,
            subdir=args.model_subdir,
            model_family=backbone.model_family if backbone_explicit or is_axis else "auto",
        )
    except DownloadError as e:
        # 基础设施失败(网络/镜像/Hub),与"模型本身不合法"区分开
        sys.exit(f"[run_eval] model download failed: {e}")
    except (FileNotFoundError, ValueError) as e:
        _reject_model(str(e))
    print(f"[run_eval] checkpoint: {ckpt_dir}")
    print(f"[run_eval] benchmark: {args.benchmark}")

    try:
        detected_family = detect_model_family(ckpt_dir)
    except ValueError as e:
        _reject_model(str(e))
    if args.model_family == "auto":
        args.model_family = detected_family
    elif args.model_family != detected_family:
        _reject_model(
            f"--model-family {args.model_family!r} does not match detected checkpoint family {detected_family!r}"
        )
    if not backbone_explicit and detected_family != backbone.model_family:
        backbone = next(item for item in BACKBONES.values() if item.model_family == detected_family)
        args.backbone = backbone.name
        model_architectures = backbone.legacy_architectures
    elif detected_family != backbone.model_family:
        _reject_model(
            f"--backbone {backbone.name!r} expects family {backbone.model_family!r}, detected {detected_family!r}"
        )
    server_python = {
        "openpi": SERVER_VENV_PY,
        "openvla_oft": OPENVLA_OFT_VENV_PY,
        "lingbot_vla_v2": LINGBOT_VLA_V2_VENV_PY,
    }[args.model_family]
    if not server_python.exists():
        sys.exit(f"[run_eval] Missing {server_python} — install the {args.model_family} server env (bash setup.sh).")
    print(f"[run_eval] model family: {args.model_family}")
    if args.server_impl == "static":
        if args.model_family != "lingbot_vla_v2":
            sys.exit("[run_eval] --server-impl static is supported only for LingBot-VLA 2.0")
        if args.workers_per_gpu < args.max_batch or args.workers_per_gpu % args.max_batch:
            sys.exit(
                "[run_eval] static batching requires --workers-per-gpu to be a positive multiple "
                f"of --max-batch (got {args.workers_per_gpu} and {args.max_batch})"
            )

    # Legality gate: reject malformed checkpoints with explicit reasons before
    # spending any GPU time (see check_model.py for what is validated).
    if args.skip_model_check:
        if args.model_family == "openvla_oft":
            args.config = args.config or "openvla_oft_libero"
        elif args.model_family == "lingbot_vla_v2":
            args.config = args.config or "lingbot-vla-v2"
        else:
            args.config = args.config or ARCHITECTURE_CONFIGS[model_architectures[0]]
        print("[run_eval] model format check SKIPPED (--skip-model-check)")
    elif args.model_family == "openvla_oft":
        if args.config is not None:
            sys.exit("[run_eval] --config is only valid for openpi checkpoints")
        check = check_openvla_oft_model(ckpt_dir)
        for w in check.warnings:
            print(f"[run_eval] model check warning: {w}")
        if not check.ok:
            print(f"[run_eval] model REJECTED by the OpenVLA-OFT format check ({len(check.errors)} problem(s)):")
            for i, problem in enumerate(check.errors, 1):
                print(f"  {i}. {problem}")
            sys.exit(3)
        args.config = check.config
        print("[run_eval] model format check passed (OpenVLA-OFT sharded checkpoint)")
    elif args.model_family == "lingbot_vla_v2":
        if args.config is not None:
            sys.exit("[run_eval] --config is only valid for openpi checkpoints")
        check = check_lingbot_vla_v2_model(ckpt_dir)
        for warning in check.warnings:
            print(f"[run_eval] model check warning: {warning}")
        if not check.ok:
            print(f"[run_eval] model REJECTED by the LingBot-VLA 2.0 format check ({len(check.errors)} problem(s)):")
            for index, problem in enumerate(check.errors, 1):
                print(f"  {index}. {problem}")
            sys.exit(3)
        contract_errors = check_lingbot_data_contract(
            ckpt_dir,
            expected_cameras=("camera_top", "camera_wrist"),
            required_joints={"end.position": 14, "effector.position": 2},
            metadata_required=False,
        )
        if contract_errors:
            print(
                f"[run_eval] model REJECTED by the LingBot LIBERO data-contract check "
                f"({len(contract_errors)} problem(s)):"
            )
            for index, problem in enumerate(contract_errors, 1):
                print(f"  {index}. {problem}")
            print(
                "  This benchmark needs a LingBot-VLA 2.0 checkpoint fine-tuned for the evaluator's "
                "two-camera, 7D LIBERO action contract; the official RoboTwin checkpoint is not interchangeable."
            )
            sys.exit(3)
        args.config = check.config
        print("[run_eval] model format check passed (LingBot-VLA 2.0 sharded checkpoint)")
    else:
        selection = check_model_for_architectures(ckpt_dir, model_architectures, args.config)
        check = selection.result
        for w in check.warnings:
            print(f"[run_eval] model check warning: {w}")
        if not check.ok:
            print(f"[run_eval] model REJECTED by the openpi format check ({len(check.errors)} problem(s)):")
            for i, problem in enumerate(check.errors, 1):
                print(f"  {i}. {problem}")
            sys.exit(3)
        args.config = selection.config
        args.backbone = selection.architecture
        print(
            f"[run_eval] model format check passed "
            f"({check.checkpoint_type} checkpoint, architecture {selection.architecture}, config {args.config})"
        )

    if is_axis:
        try:
            from axis_backend import run as run_axis

            return_code = run_axis(
                args,
                ckpt_dir,
                gpus,
                AXIS_VENV_PY,
                start_servers=start_servers,
                wait_for_servers=wait_for_servers,
                stop_servers=stop_servers,
            )
        except (FileNotFoundError, RuntimeError, TimeoutError, ValueError) as e:
            sys.exit(f"[run_eval] AXIS evaluation rejected: {e}")
        if return_code:
            sys.exit(return_code)
        return

    lingbot_norm_stats = None
    lingbot_robot_config_root = None
    lingbot_data_contract = None
    lingbot_runtime_metadata = None
    qwen3_vl_path = None
    if args.model_family == "lingbot_vla_v2":
        lingbot_norm_stats = pathlib.Path(args.lingbot_norm_stats).expanduser().resolve()
        lingbot_data_contract = pathlib.Path(args.lingbot_data_contract).expanduser().resolve()
        lingbot_robot_config_root = DEFAULT_LINGBOT_ROBOT_CONFIG_ROOT
        qwen3_vl_path = pathlib.Path(args.qwen3_vl_path).expanduser().resolve()
        required = (
            lingbot_norm_stats,
            lingbot_data_contract,
            lingbot_robot_config_root / "libero.yaml",
            qwen3_vl_path / "config.json",
        )
        if missing := [str(path) for path in required if not path.exists()]:
            sys.exit("[run_eval] LingBot-VLA 2.0 LIBERO runtime is incomplete; missing: " + ", ".join(missing))
        try:
            lingbot_runtime_metadata = runtime_contract_metadata(lingbot_data_contract, lingbot_norm_stats)
        except ValueError as exc:
            sys.exit(f"[run_eval] LingBot-VLA 2.0 LIBERO runtime contract is invalid: {exc}")

    model_tag = pathlib.Path(str(args.model).rstrip("/")).name.replace("/", "_")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        pathlib.Path(args.output_dir)
        if args.output_dir
        else (VALIDATOR_ROOT / "eval_runs" / f"{stamp}_{bench.name}_{model_tag}")
    )
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    print(f"[run_eval] output dir: {out_dir}")

    # Fetch benchmark assets if needed (LIBERO-Pro bddl/init downloads) and
    # write the variant's LIBERO config before any client import.
    bench.prepare(suites)

    # Warm up the libero package once (avoids a first-import config race across
    # parallel workers), fail fast on suite names it does not register, and
    # report each suite's task count (suites range from 10 tasks in the base
    # benchmark to 2591 in LIBERO-plus). Suite instantiation prints chatty
    # "[info] using task orders ..." lines, hence the stdout redirect; the
    # counts json is the probe's only stdout line.
    check_suites = (
        "import contextlib, io, json, sys\n"
        "from libero.libero import benchmark\n"
        "d = benchmark.get_benchmark_dict()\n"
        "missing = [s for s in sys.argv[1:] if s not in d]\n"
        "assert not missing, 'suites not registered in this benchmark: %s' % missing\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    counts = {s: d[s]().n_tasks for s in sys.argv[1:]}\n"
        "print(json.dumps(counts))\n"
    )
    probe = subprocess.run(
        [str(CLIENT_VENV_PY), "-c", check_suites, *suites],
        env={**_base_env(), **bench.client_env()},
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        sys.exit(f"[run_eval] benchmark '{bench.name}' rejected the requested suites:\n{probe.stderr.strip()}")
    suite_n_tasks = json.loads(probe.stdout.strip().splitlines()[-1])
    if task_ids is not None:
        out_of_range = [(s, t) for s in suites for t in task_ids if not 0 <= t < suite_n_tasks[s]]
        if out_of_range:
            sys.exit(f"[run_eval] task ids out of range: {out_of_range} (suite sizes: {suite_n_tasks})")

    protocol = bench.protocol_request(suites, suite_n_tasks, task_ids, args.num_trials)
    if protocol is not None:
        if protocol["official_request"]:
            print(
                f"[run_eval] protocol: {protocol['name']} OFFICIAL "
                f"({protocol['expected_task_count']} tasks x {protocol['expected_trials_per_task']} trial)"
            )
        else:
            print(f"[run_eval] protocol: NON-OFFICIAL DEVELOPMENT RUN; {'; '.join(protocol['deviations'])}")

    if args.init_seed is not None:
        seeded_inits = _required_seeded_inits(args.num_trials)
        _init_log(
            f"ensuring seeded init states exist: seed={args.init_seed} "
            f"tasks={sum(suite_n_tasks[suite] for suite in suites)} trials_per_task={args.num_trials} "
            f"seeded_inits_per_task={seeded_inits} workers_per_gpu={args.init_workers_per_gpu}"
        )
        if seeded_inits == 0:
            _init_log("no seeded episodes are needed for this trial count; using official init states only")
        else:
            t_gen = time.time()
            root = ensure_seeded_init_states(
                args.init_seed,
                seeded_inits,
                suites,
                suite_n_tasks,
                gpus,
                args.init_workers_per_gpu,
                bench,
                out_dir / "logs",
            )
            args.init_states_root = str(root)
            args.init_states_mix = True
            _init_log(
                f"init states ready in {time.time() - t_gen:.0f}s; trials are split "
                f"50/50 official + seed {args.init_seed} ({root})"
            )

    static_schedule_profile = static_scheduling_profile(bench.name) if args.server_impl == "static" else None
    specs = []
    for suite in suites:
        for task_id in task_ids if task_ids is not None else range(suite_n_tasks[suite]):
            max_steps = bench.resolve_max_steps(suite)
            specs.append(
                TaskSpec(
                    suite=suite,
                    task_id=task_id,
                    max_steps=max_steps,
                    scheduling_cost=(
                        static_scheduling_cost(bench.name, suite, max_steps) if args.server_impl == "static" else None
                    ),
                )
            )
    # Highest estimated cost first (stable sort keeps task_id order within a
    # suite). Static LingBot uses a fixed benchmark profile because max_steps
    # alone is a poor runtime estimate; other paths retain the old max-step key.
    specs.sort(key=lambda spec: spec.dispatch_cost, reverse=True)

    # Build the lane map from the complete task set, before --resume removes
    # finished entries. This preserves task -> GPU/cohort/lane identity when a
    # stopped run is resumed in the same output directory.
    static_lane_plan = (
        partition_static_lanes(specs, len(gpus) * args.workers_per_gpu) if args.server_impl == "static" else None
    )

    progress_file = pathlib.Path(args.progress_file).resolve() if args.progress_file else None

    results: dict = {}
    if args.resume:
        specs, results = split_resumable(specs, out_dir)
        print(f"[run_eval] resume: {len(results)} tasks already have results in {out_dir}, {len(specs)} to run")
    suite_task_totals = {suite: sum(spec.suite == suite for spec in specs) for suite in suites}

    task_q: queue.Queue[TaskSpec] | None = None
    static_lane_queues: list[queue.Queue[TaskSpec]] | None = None
    if args.server_impl == "static":
        assert static_lane_plan is not None
        todo_names = {spec.name for spec in specs}
        lane_specs = [[spec for spec in assignments if spec.name in todo_names] for assignments in static_lane_plan]
        static_lane_queues = []
        for assignments in lane_specs:
            lane_q: queue.Queue[TaskSpec] = queue.Queue()
            for spec in assignments:
                lane_q.put(spec)
            static_lane_queues.append(lane_q)
        print(
            f"[run_eval] static task map: {len(static_lane_queues)} fixed lanes, "
            f"batch_size={args.max_batch}, cohorts_per_gpu={args.workers_per_gpu // args.max_batch}, "
            f"scheduling_profile={static_schedule_profile}"
        )
    else:
        task_q = queue.Queue()
        for spec in specs:
            task_q.put(spec)
    print(
        f"[run_eval] {len(specs)} tasks ({len(suites)} suites, "
        f"{'ids ' + str(task_ids) if task_ids is not None else 'all tasks'}) "
        f"on {len(gpus)} GPUs x {args.workers_per_gpu} workers"
    )

    t_start = time.time()
    servers = start_servers(
        gpus,
        args.base_port,
        args.config,
        ckpt_dir,
        out_dir / "logs",
        args.mem_fraction,
        server_impl=args.server_impl,
        max_batch=args.max_batch,
        model_family=args.model_family,
        seed=args.seed,
        lingbot_norm_stats=lingbot_norm_stats,
        lingbot_robot_config_root=lingbot_robot_config_root,
        lingbot_data_contract=lingbot_data_contract,
        lingbot_compile=args.lingbot_compile,
        qwen3_vl_path=qwen3_vl_path,
        lane_count=args.workers_per_gpu,
    )

    try:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
        print(f"[run_eval] waiting for {len(servers)} servers to load the model ...")
        wait_for_servers(servers, args.server_timeout)

        lock = threading.Lock()
        progress = {
            "done": 0,
            "total": len(specs),
            "suite_done": {suite: 0 for suite in suites},
            "suite_total": suite_task_totals,
            "suites_done": 0,
            "suites_total": len(suites),
            "episodes_done": 0,
            "episodes_total": len(specs) * args.num_trials,
            "progress_file": progress_file,
        }
        emit_progress_event(
            progress_file,
            {
                "suites_done": 0,
                "suites_total": len(suites),
                "episodes_done": 0,
                "episodes_total": len(specs) * args.num_trials,
            },
        )
        threads = []
        for server_index, server in enumerate(servers):
            for lane_id in range(max(1, args.workers_per_gpu)):
                lane_q = (
                    static_lane_queues[
                        interleaved_static_lane_index(server_index, lane_id, len(servers), args.max_batch)
                    ]
                    if static_lane_queues is not None
                    else task_q
                )
                assert lane_q is not None
                threads.append(
                    threading.Thread(
                        target=worker_loop,
                        args=(server, lane_q, args, bench, out_dir, results, lock, progress, lane_id),
                        daemon=True,
                    )
                )
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    finally:
        stop_servers(servers)

    task_groups = {s: g for s in suites if (g := bench.task_groups(s)) is not None}
    summary = summarize(
        results,
        {
            "model": str(args.model),
            "model_family": args.model_family,
            "backbone": args.backbone,
            "commit_id": args.commit_id,
            "evaluator_source_git_commit": args.evaluator_source_git_commit,
            "checkpoint_dir": str(ckpt_dir),
            "benchmark": bench.name,
            "prompt_source": bench.prompt_source,
            "config": args.config,
            "num_trials_per_task": args.num_trials,
            "seed": args.seed,
            "init_states_root": args.init_states_root,
            "init_seed": args.init_seed,
            "init_states_mix": args.init_states_mix,
            "gpus": gpus,
            "workers_per_gpu": args.workers_per_gpu,
            "policy_rng_mode": POLICY_RNG_MODE if args.model_family == "lingbot_vla_v2" else None,
            "policy_batch_mode": POLICY_BATCH_MODE_STATIC if args.server_impl == "static" else None,
            "server_impl": args.server_impl,
            "static_batch_size": args.max_batch if args.server_impl == "static" else None,
            "static_scheduling_profile": static_schedule_profile,
            "deterministic_algorithms": args.server_impl == "static",
            "wall_time_s": round(time.time() - t_start, 1),
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
        },
        task_groups=task_groups or None,
    )
    if protocol is not None:
        failed_tasks = sum(r.get("status") != "ok" for r in results.values())
        protocol["completed_task_count"] = len(results)
        protocol["completed_episode_count"] = summary["total_episodes"]
        protocol["failed_task_count"] = failed_tasks
        protocol["official_result"] = bool(
            protocol["official_request"]
            and len(results) == protocol["expected_task_count"]
            and summary["total_episodes"] == protocol["expected_task_count"] * protocol["expected_trials_per_task"]
            and failed_tasks == 0
        )
        if protocol["official_request"] and not protocol["official_result"]:
            protocol["deviations"].append("evaluation did not complete every official task successfully")
        summary["evaluation_protocol"] = protocol
    if lingbot_runtime_metadata is not None:
        summary["lingbot_runtime"] = lingbot_runtime_metadata
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print_report(summary)
    print(f"[run_eval] summary written to {out_dir / 'summary.json'}")

    if any(r["status"] != "ok" for r in results.values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
