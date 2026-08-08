"""Parallel LIBERO / LIBERO-Pro evaluation orchestrator for multi-GPU machines.

Input: a model reference — a local openpi checkpoint dir, a Hugging Face repo id
(`user/repo`), or a HF URL (`https://huggingface.co/user/repo`).

What it does:
  1. Resolves the model to a local checkpoint directory. HF references are
     downloaded pinned to --commit-id (mandatory), and the download cache is
     keyed by that commit — a re-submission to the same repo can never be
     confused with a previously evaluated version.
  2. Prepares the selected benchmark (--benchmark libero|libero_pro|libero_plus;
     see benchmarks.py — LIBERO-Pro/LIBERO-plus assets are downloaded on first
     use).
  3. Spawns one openpi policy server per GPU (CUDA_VISIBLE_DEVICES isolation).
  4. Builds the task list (suites x task_ids, default all tasks per suite),
     dispatches tasks dynamically to per-GPU workers (each worker runs
     eval_task.py in the LIBERO client venv, rendering via EGL on its own GPU).
  5. Aggregates per-task JSONs into summary.json and prints a report (per suite,
     plus per perturbation dimension for LIBERO-plus).

Run from the validator repo root (env managed by uv, see pyproject.toml):
    uv run libero_eval/run_eval.py --model <path-or-hf-repo> --commit-id <hf-commit-sha> [options]
"""

import argparse
import dataclasses
import datetime
import fcntl
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

from benchmarks import BENCHMARKS, get_benchmark
from check_model import (
    ARCHITECTURE_CONFIGS,
    MODEL_FAMILIES,
    check_openvla_oft_model,
    check_model_for_architectures,
    detect_model_family,
    parse_model_architectures,
)
from download import COMMIT_HASH_RE, DEFAULT_STRATEGIES, DownloadError, download_model, parse_strategies
from paths import (
    CLIENT_VENV_PY,
    OPENPI_DIR,
    OPENVLA_OFT_DIR,
    OPENVLA_OFT_VENV_PY,
    SERVER_VENV_PY,
    VALIDATOR_ROOT,
)

EVAL_TASK_SCRIPT = pathlib.Path(__file__).resolve().parent / "eval_task.py"


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
) -> pathlib.Path:
    """Resolve a model reference (local path / HF repo id / HF URL) to a local checkpoint dir.

    HF references are downloaded pinned to `commit_id` (full 40-hex sha), and
    the download dir is keyed by the commit (same layout as benchmark_worker:
    `<user>__<repo>@<commit12>`) — different commits never share a cache dir.
    Local paths are used as-is; commit_id is only recorded in summary.json.
    """
    # Resolve to an absolute path: the policy server subprocess runs with its
    # cwd inside the openpi checkout, so a relative --model would break there.
    local = pathlib.Path(model).expanduser()
    if local.exists():
        return _find_checkpoint_root(local.resolve())

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
    local_dir = (download_dir / f"{repo_id.replace('/', '__')}@{commit_id[:12]}").resolve()
    download_model(repo_id, local_dir, revision=commit_id, strategies=strategies)
    return _find_checkpoint_root(local_dir)


def _find_checkpoint_root(path: pathlib.Path) -> pathlib.Path:
    """Find the directory that actually contains a supported checkpoint.

    Accepts either a JAX checkpoint (params/ dir) or a PyTorch checkpoint
    (model.safetensors), possibly nested one or two levels deep (e.g. HF repos
    laid out as `checkpoints/<step>/params`).
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
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No supported checkpoint found under {path} (expected params/, model.safetensors, "
        "or model.safetensors.index.json)."
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
                s.bind(("localhost", port))
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
    max_batch: int = 8,
    model_family: str = "openpi",
    seed: int = 7,
) -> list[PolicyServer]:
    ports = _find_free_ports(base_port, len(gpus))
    # Persistent XLA compilation cache: the pi0.5 sample_actions JIT takes ~20s
    # and is identical across servers and across runs (same config/shapes), so
    # after the first run every server skips it.
    jax_cache_dir = VALIDATOR_ROOT / ".cache" / "jax_compilation"
    jax_cache_dir.mkdir(parents=True, exist_ok=True)
    is_pytorch = model_family == "openpi" and (ckpt_dir / "model.safetensors").is_file()
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
            "JAX_COMPILATION_CACHE_DIR": str(jax_cache_dir),
        }
        if is_pytorch:
            # serve_policy imports jax even on the PyTorch path; keep it off the
            # GPU so it cannot preallocate mem_fraction of VRAM ahead of torch.
            env["JAX_PLATFORMS"] = "cpu"
        if model_family == "openvla_oft":
            # TensorFlow is used only for the official JPEG/Lanczos + center-crop
            # preprocessing.  Its default is "all host cores per process"; with
            # one server per GPU that causes catastrophic CPU oversubscription.
            env.update(
                {
                    "TF_NUM_INTRAOP_THREADS": "2",
                    "TF_NUM_INTEROP_THREADS": "1",
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                    "OPENVLA_TORCH_THREADS": "2",
                }
            )
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
        log_f = open(log_path, "w")
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


def wait_for_servers(servers: list[PolicyServer], timeout_s: float) -> None:
    """Block until every server accepts TCP connections (model loaded), or raise."""
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
                with socket.create_connection(("localhost", port), timeout=1):
                    pass
                # Port accepts connections AND our process is still alive -> it is
                # really our server listening (a bind failure would have killed it).
                if s.proc.poll() is None:
                    print(f"[run_eval] server ready: gpu={s.gpu} port={port}")
                    del pending[port]
            except OSError:
                pass
        if pending:
            if time.time() > deadline:
                raise TimeoutError(f"Servers not ready after {timeout_s}s: ports {sorted(pending)}")
            time.sleep(2)


def stop_servers(servers: list[PolicyServer]) -> None:
    for s in servers:
        if s.proc.poll() is None:
            s.proc.terminate()
    deadline = time.time() + 10
    for s in servers:
        while s.proc.poll() is None and time.time() < deadline:
            time.sleep(0.2)
        if s.proc.poll() is None:
            s.proc.kill()


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


def ensure_seeded_init_states(
    seed, num_inits, suites, suite_n_tasks, gpus, init_workers_per_gpu, bench, log_dir
):
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
        env = {**_base_env(), "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(gpu), **bench.client_env()}
        log_path = log_dir / f"gen_init_seed{seed}_gpu{gpu}_worker{worker_id}.log"
        log_f = open(log_path, "w")
        procs.append(
            (gpu, worker_id, subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT), log_f)
        )

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

    @property
    def name(self) -> str:
        return f"{self.suite}_task{self.task_id:02d}"


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


def worker_loop(
    server: PolicyServer,
    task_q: "queue.Queue[TaskSpec]",
    args,
    bench,
    out_dir: pathlib.Path,
    results: dict,
    lock: threading.Lock,
    progress: dict,
) -> None:
    while True:
        try:
            spec = task_q.get_nowait()
        except queue.Empty:
            return

        result_json = out_dir / "results" / f"{spec.name}.json"
        log_path = out_dir / "logs" / f"{spec.name}.log"
        cmd = [
            str(CLIENT_VENV_PY),
            str(EVAL_TASK_SCRIPT),
            "--host",
            "localhost",
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
        if args.init_states_root:
            cmd += ["--init-states-root", str(args.init_states_root)]
            if args.init_states_mix:
                cmd += ["--init-states-mix"]
        if args.save_videos > 0:
            cmd += ["--video-dir", str(out_dir / "videos"), "--save-videos", str(args.save_videos)]

        env = {
            **_base_env(),
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": str(server.gpu),
            **bench.client_env(),
        }

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
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Local checkpoint dir, HF repo id (user/repo), or HF URL")
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
        "--benchmark",
        default="libero",
        choices=sorted(BENCHMARKS),
        help="Which benchmark to evaluate on (default: libero)",
    )
    parser.add_argument(
        "--model-architectures",
        default="pi0.5",
        metavar="ARCH[,ARCH...]",
        help="Accepted model architectures (default: pi0.5; e.g. pi0 or pi0,pi0.5; π spellings also work)",
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
        "--num-trials",
        type=int,
        default=None,
        help="Trials per task (default: the benchmark's protocol — 10 for libero/libero_pro "
        "[official LIBERO: 50], 1 for libero_plus)",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=3,
        help="Concurrent eval clients per GPU sharing one policy server (default 3: measured "
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
        choices=("upstream", "batched"),
        default="upstream",
        help="Policy server implementation: 'upstream' = openpi's serve_policy.py (batch 1); "
        "'batched' = serve_policy_batched.py, which greedily batches whatever requests are "
        "queued (nobody waits for a batch to fill) — pair with a higher --workers-per-gpu",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=8,
        help="Upper bound on the dynamic batch size (batched server only); padded to powers of 2",
    )
    parser.add_argument("--base-port", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=7)
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
    args = parser.parse_args()

    if args.init_workers_per_gpu < 1:
        parser.error("--init-workers-per-gpu must be at least 1")

    try:
        model_architectures = parse_model_architectures(args.model_architectures)
    except ValueError as e:
        parser.error(str(e))

    for py, what in [(CLIENT_VENV_PY, "LIBERO client venv")]:
        if not py.exists():
            sys.exit(f"[run_eval] Missing {py} — install the {what} first (bash setup.sh; see README).")

    # Preflight: a hung NVIDIA driver makes every CUDA/EGL process enter an
    # unkillable D state. Detect it up front instead of spawning 8 servers into it.
    try:
        subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=15, check=True)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        sys.exit(
            f"[run_eval] GPU preflight failed ({type(e).__name__}): nvidia-smi is not responding.\n"
            "The NVIDIA driver may be hung (D-state processes). A reboot or driver reload is "
            "required before evaluation can run."
        )

    gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    try:
        # Keep these file objects alive until main exits. The OS releases the
        # flock automatically on normal, exceptional, or signalled exit.
        _gpu_locks = acquire_gpu_locks(gpus)
    except (OSError, ValueError, RuntimeError) as e:
        sys.exit(f"[run_eval] GPU reservation failed: {e}")

    bench = get_benchmark(args.benchmark)
    suites = [s.strip() for s in args.suites.split(",") if s.strip()] if args.suites else list(bench.default_suites)
    task_ids = [int(t) for t in args.task_ids.split(",")] if args.task_ids else None
    if args.num_trials is None:
        args.num_trials = bench.default_num_trials

    args.init_states_mix = False
    if args.init_states_root and args.init_seed is not None:
        sys.exit("[run_eval] --init-seed and --init-states-root are mutually exclusive")
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
        ckpt_dir = resolve_model(args.model, pathlib.Path(args.download_dir), strategies, commit_id=args.commit_id)
    except DownloadError as e:
        # 基础设施失败(网络/镜像/Hub),与"模型本身不合法"区分开
        sys.exit(f"[run_eval] model download failed: {e}")
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[run_eval] model REJECTED: {e}")
    print(f"[run_eval] checkpoint: {ckpt_dir}")
    print(f"[run_eval] benchmark: {bench.name}")

    try:
        detected_family = detect_model_family(ckpt_dir)
    except ValueError as e:
        sys.exit(f"[run_eval] model REJECTED: {e}")
    if args.model_family == "auto":
        args.model_family = detected_family
    elif args.model_family != detected_family:
        sys.exit(
            f"[run_eval] model REJECTED: --model-family {args.model_family!r} does not match "
            f"detected checkpoint family {detected_family!r}"
        )
    server_python = OPENVLA_OFT_VENV_PY if args.model_family == "openvla_oft" else SERVER_VENV_PY
    if not server_python.exists():
        sys.exit(f"[run_eval] Missing {server_python} — install the {args.model_family} server env (bash setup.sh).")
    print(f"[run_eval] model family: {args.model_family}")

    # Legality gate: reject malformed checkpoints with explicit reasons before
    # spending any GPU time (see check_model.py for what is validated).
    if args.skip_model_check:
        args.config = args.config or (
            "openvla_oft_libero"
            if args.model_family == "openvla_oft"
            else ARCHITECTURE_CONFIGS[model_architectures[0]]
        )
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
        print(
            f"[run_eval] model format check passed "
            f"({check.checkpoint_type} checkpoint, architecture {selection.architecture}, config {args.config})"
        )

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
            print(
                f"[run_eval] protocol: NON-OFFICIAL DEVELOPMENT RUN; "
                f"{'; '.join(protocol['deviations'])}"
            )

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

    specs = [
        TaskSpec(suite=s, task_id=t, max_steps=bench.resolve_max_steps(s))
        for s in suites
        for t in (task_ids if task_ids is not None else range(suite_n_tasks[s]))
    ]
    # Longest tasks first (stable sort keeps task_id order within a suite): a
    # 520-step libero_10 task dispatched last would stretch the tail by minutes
    # while every other worker sits idle.
    specs.sort(key=lambda sp: sp.max_steps, reverse=True)

    progress_file = pathlib.Path(args.progress_file).resolve() if args.progress_file else None

    results: dict = {}
    if args.resume:
        specs, results = split_resumable(specs, out_dir)
        print(f"[run_eval] resume: {len(results)} tasks already have results in {out_dir}, {len(specs)} to run")
    suite_task_totals = {suite: sum(spec.suite == suite for spec in specs) for suite in suites}

    task_q: queue.Queue[TaskSpec] = queue.Queue()
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
        threads = [
            threading.Thread(
                target=worker_loop, args=(s, task_q, args, bench, out_dir, results, lock, progress), daemon=True
            )
            for s in servers
            for _ in range(max(1, args.workers_per_gpu))
        ]
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
            "commit_id": args.commit_id,
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
            "server_impl": args.server_impl,
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
            and summary["total_episodes"]
            == protocol["expected_task_count"] * protocol["expected_trials_per_task"]
            and failed_tasks == 0
        )
        if protocol["official_request"] and not protocol["official_result"]:
            protocol["deviations"].append("evaluation did not complete every official task successfully")
        summary["evaluation_protocol"] = protocol
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print_report(summary)
    print(f"[run_eval] summary written to {out_dir / 'summary.json'}")

    if any(r["status"] != "ok" for r in results.values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
