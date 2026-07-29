"""
Benchmark Worker — 后端评测队列与本仓库评测流水线之间的编排层。
Orchestration layer bridging the backend benchmark API and this repo's
evaluation pipeline.

职责(与评测执行严格隔离,本进程不运行任何评测逻辑):
  1. 用 public key 轮询 GET /api/v1/benchmark/queue 获取待评测任务
  2. 维护本地持久化任务队列(state.json:跨轮询、跨重启去重)
  3. 逐个派发:下载模型(锁定 hf_commit)→ 起 libero_eval/run_eval.py
     子进程 → 解析 summary.json
  4. 用 admin key POST /api/v1/benchmark/task/{task_id}/score 回传分数,失败持续重试

GPU、policy server、MuJoCo 等评测细节全部在 run_eval.py 子进程内;
本进程只做 HTTP I/O 与进程管理(模型下载复用 huggingface_hub)。
"""

import argparse
import datetime
import json
import logging
import math
import os
import pathlib
import queue
import re
import signal
import subprocess
import sys
import threading
import time

VALIDATOR_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VALIDATOR_ROOT))

from benchmark_worker.backend_client import BackendClient, BackendError
from benchmark_worker.scoring import (
    build_score_payload,
    prepare_submit_payload,
)
from benchmark_worker.profiles import PROFILES, get_profile
from benchmark_worker.state import StateStore
from libero_eval.download import COMMIT_HASH_RE, DownloadError, download_model, parse_strategies

RUN_EVAL = VALIDATOR_ROOT / "libero_eval" / "run_eval.py"

_REPO_ID_RE = re.compile(r"^[\w][\w.-]*/[\w][\w.-]*$")
_SUITE_RE = re.compile(r"^[a-z0-9_]+$")

stop_event = threading.Event()
_inflight_lock = threading.Lock()
_inflight: set[str] = set()  # task_id 正被 worker 线程处理(评测或提交中)

_BENCHMARK_PROTOCOL_REVISIONS = {
    "libero_plus": "libero_plus_official_v1",
}

logger = logging.getLogger("benchmark_worker")


def _protocol_revision(benchmark: str | None) -> str | None:
    return _BENCHMARK_PROTOCOL_REVISIONS.get(benchmark or "")


def _model_label(d: dict | None) -> str:
    """任务或评分 payload 里的模型标识,供日志定位:hf_repo_id@commit 前 12 位。

    task_id 是后端生成的不透明串,单看它不知道评的是哪个模型;所有任务级
    日志都应带上这个标签,读日志的人才能对上"哪个 repo 的哪次提交"。
    """
    d = d or {}
    repo = d.get("hf_repo_id") or "?"
    commit = (d.get("hf_commit") or "")[:12] or "?"
    return f"{repo}@{commit}"


def _setup_logger() -> None:
    """控制台 + 按天滚动的文件日志(logs/,gitignore)。"""
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    log_dir = VALIDATOR_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    date_str = datetime.date.today().isoformat()
    file_handler = logging.FileHandler(log_dir / f"benchmark_worker-{date_str}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)


class EvalInterrupted(Exception):
    """收到退出信号,当前评测被中止(任务会重新排队,不上报失败)。"""


class EvalInfrastructureError(Exception):
    """本机评测基础设施暂时不可用,任务应重新排队且不上报失败。"""


# ----------------------------------------------------------------------------
# 模型解析
# ----------------------------------------------------------------------------
def resolve_model(
    task: dict, download_dir: pathlib.Path, allow_local: bool
) -> tuple[str | None, str | None, pathlib.Path]:
    """校验队列任务的 hf_repo_id + hf_commit,计算本地模型目录;不执行下载。

    hf_commit 必填(完整 40 位 hex):没有 commit 就无法判定"重新提交同一
    repo"与"已评测版本"是否同一份内容,评测结果也无法锚定到唯一提交,
    直接拒绝并把原因上报后端(miner 可见)。

    返回 (ref, revision, local_dir);ref 为 None 表示 local_dir 是现成的
    本地模型目录(--allow-local-model),无须下载。
    """
    ref = (task.get("hf_repo_id") or "").strip()
    if allow_local:
        local = pathlib.Path(ref).expanduser()
        if local.exists():
            return None, None, local.resolve()
    if not _REPO_ID_RE.match(ref):
        raise ValueError(f"invalid hf_repo_id: {ref!r}")
    revision = (task.get("hf_commit") or "").strip()
    if not revision:
        raise ValueError(
            "task has no hf_commit: evaluation must be pinned to an exact commit "
            "(re-submit with the model's HF commit hash)"
        )
    if not COMMIT_HASH_RE.match(revision):
        raise ValueError(f"invalid hf_commit: {revision!r} (expected the full 40-char hex commit hash)")

    tag = f"{ref.replace('/', '__')}@{revision[:12]}"
    return ref, revision, (download_dir / tag).resolve()


def download_with_retry(ref: str, revision: str | None, model_dir: pathlib.Path, args) -> None:
    """下载模型,对失败做有上限的原地重试。

    下载失败是网络/镜像侧的暂时性故障(代理上游抖动时三种策略会同时 SSL
    失败),不构成对模型的评测结论;这里重试若干次,耗尽后抛 DownloadError
    由调用方把任务重新排队——绝不能走"失败报告"路径。
    """
    for attempt in range(args.download_retries):
        try:
            download_model(ref, model_dir, revision=revision, strategies=args.download_strategies, log=logger.info)
            return
        except DownloadError as e:
            if attempt + 1 >= args.download_retries:
                raise
            wait = min(300, 30 * 2**attempt)
            logger.warning(f"download failed (attempt {attempt + 1}/{args.download_retries}): {e}; retrying in {wait}s")
            if stop_event.wait(wait):
                raise EvalInterrupted() from None


# ----------------------------------------------------------------------------
# 评测子进程
# ----------------------------------------------------------------------------
def _terminate(proc: subprocess.Popen) -> None:
    # 先礼后兵:SIGTERM 让 run_eval 自己清理 policy server,超时再杀整个会话
    # (start_new_session=True 保证 pid 即 pgid)。
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


_RETRYABLE_EVAL_LOG_MARKERS = (
    "gpu reservation failed",
    "cuda out of memory",
    "cuda_error_out_of_memory",
    "resource_exhausted",
    "failed to allocate memory",
    # MuJoCo/robosuite could not allocate an EGL render target. This is a
    # validator-side GPU resource failure, not evidence about model quality.
    "offscreen framebuffer is not complete",
    "egl_bad_alloc",
    "egl_not_initialized",
)


def _has_retryable_infrastructure_error(log_text: str) -> bool:
    lowered = log_text.lower()
    return any(marker in lowered for marker in _RETRYABLE_EVAL_LOG_MARKERS)


def _failed_task_infrastructure_detail(out_dir: pathlib.Path, failed_tasks: list[str]) -> str:
    """Return concise evidence when a failed task log contains an infra error."""
    matches = []
    for task_name in failed_tasks:
        log_path = out_dir / "logs" / f"{task_name}.log"
        try:
            log_text = log_path.read_text(errors="replace")
        except OSError:
            continue
        if not _has_retryable_infrastructure_error(log_text):
            continue
        tail = "\n".join(log_text.splitlines()[-15:])
        matches.append(f"{task_name} ({log_path}):\n{tail}")
    return "\n\n".join(matches)


def run_evaluation(task: dict, model_dir: pathlib.Path, out_dir: pathlib.Path, args, init_seed: int | None = None):
    """运行一次 run_eval.py,返回 (summary | None, error_str)。

    init_seed 非空时启用 init states 混合随机化(run_eval --init-seed:
    每 task 一半 trial 用官方初始状态、一半用该 seed 重采样的初始状态)。
    收到退出信号时中止子进程并抛 EvalInterrupted。
    """
    selected_benchmark = getattr(args, "benchmark", "libero")
    profile = get_profile(selected_benchmark)
    benchmark = profile.runtime_benchmark
    commit_id = (task.get("hf_commit") or "").strip() or "local"
    cmd = [
        sys.executable,
        str(RUN_EVAL),
        "--model",
        str(model_dir),
        "--commit-id",
        commit_id,
        "--benchmark",
        benchmark,
        "--model-architectures",
        getattr(args, "model_architectures", "pi0.5"),
        "--num-trials",
        str(args.num_trials),
        "--gpus",
        args.gpus,
        "--workers-per-gpu",
        str(args.workers_per_gpu),
        "--init-workers-per-gpu",
        str(getattr(args, "init_workers_per_gpu", 4)),
        "--server-impl",
        args.server_impl,
        "--output-dir",
        str(out_dir),
    ]
    if args.eval_config:
        cmd += ["--config", args.eval_config]
    if args.task_ids:
        cmd += ["--task-ids", args.task_ids]
    # LIBERO-Plus variants ship perturbation-specific init states; replacing
    # them changes the benchmark definition and run_eval correctly rejects it.
    if init_seed is not None and benchmark != "libero_plus":
        cmd += ["--init-seed", str(init_seed)]

    log_path = out_dir / "run_eval.log"
    logger.info(
        f"Evaluation starts with command `{' '.join(cmd)}`, "
        f"benchmark={selected_benchmark} runtime_benchmark={benchmark}, "
        f"init_seed={init_seed if init_seed is not None else '(off)'}, find log at {log_path}"
    )
    with open(log_path, "w") as log_f:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        deadline = time.time() + args.eval_timeout
        while proc.poll() is None:
            if stop_event.is_set():
                _terminate(proc)
                raise EvalInterrupted()
            if time.time() > deadline:
                _terminate(proc)
                return None, f"evaluation timed out after {args.eval_timeout:.0f}s"
            time.sleep(5)

    # SIGTERM 与子进程自行退出可能发生在同一个轮询间隔内。停机意图必须
    # 优先于刚结束的子进程结果,否则会把中断/OOM 当成最终模型结论提交。
    if stop_event.is_set():
        raise EvalInterrupted()

    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        log_text = log_path.read_text(errors="replace")
        if proc.returncode == 3:
            # 模型合法性检查未通过(libero_eval/check_model.py):把拒绝理由
            # 带回后端,miner 能直接看到原因。
            tail = "\n".join(log_text.splitlines()[-15:])
            return None, f"model rejected by pre-eval check:\n{tail}"
        if _has_retryable_infrastructure_error(log_text):
            tail = "\n".join(log_text.splitlines()[-15:])
            raise EvalInfrastructureError(f"evaluation infrastructure failed:\n{tail}")
        return None, f"run_eval exited {proc.returncode} without summary.json (see {log_path})"
    summary = json.loads(summary_path.read_text())
    if selected_benchmark == "libero_plus":
        protocol = summary.get("evaluation_protocol") or {}
        if not protocol.get("official_result"):
            deviations = "; ".join(protocol.get("deviations") or ["missing official protocol metadata"])
            raise EvalInfrastructureError(f"refusing to score non-official LIBERO-Plus result: {deviations}")
    if proc.returncode == 2:
        failed = sorted(n for n, r in summary.get("tasks", {}).items() if r.get("status") != "ok")
        # run_eval writes summary.json before exiting 2. Inspect the individual
        # task logs as well: infrastructure failures must requeue the benchmark,
        # never turn a validator-side EGL/CUDA problem into a partial model score.
        infra_detail = _failed_task_infrastructure_detail(out_dir, failed)
        if infra_detail:
            raise EvalInfrastructureError(
                f"{len(failed)} evaluation task(s) failed because infrastructure was unavailable:\n{infra_detail}"
            )
        # Non-infrastructure task errors remain visible as partial results.
        return summary, f"{len(failed)} task(s) failed after retries: {', '.join(failed[:10])}"
    if proc.returncode != 0:
        return summary, f"run_eval exited {proc.returncode} (see {log_path})"
    return summary, ""


# ----------------------------------------------------------------------------
# 提交
# ----------------------------------------------------------------------------
_SUBMIT_RETRY_BASE_S = 60
_SUBMIT_RETRY_MAX_S = 30 * 60


def submit_retry_due(entry: dict, now: datetime.datetime | None = None) -> bool:
    """账本中的提交退避是否到期;缺失/损坏时间按可重试处理。"""
    raw = entry.get("next_submit_at")
    if not raw:
        return True
    try:
        due_at = datetime.datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return True
    if due_at.tzinfo is None:
        due_at = due_at.replace(tzinfo=datetime.timezone.utc)
    now = now or datetime.datetime.now().astimezone()
    return now >= due_at


def _schedule_submit_retry(store: StateStore, task_id: str, error: str) -> tuple[int, str]:
    entry = store.get(task_id) or {}
    attempts = int(entry.get("submit_attempts") or 0) + 1
    delay = min(_SUBMIT_RETRY_MAX_S, _SUBMIT_RETRY_BASE_S * 2 ** min(attempts - 1, 10))
    next_at = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=delay)
    next_at_text = next_at.isoformat(timespec="seconds")
    store.update(
        task_id,
        submit_attempts=attempts,
        submit_error=error,
        next_submit_at=next_at_text,
    )
    return delay, next_at_text


def _same_number(left, right) -> bool:
    """比较 JSON 数值,拒绝 bool/NaN/Infinity 并容忍浮点序列化尾差。"""
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left_value)
        and math.isfinite(right_value)
        and math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=1e-9)
    )


def remote_score_matches(task_id: str, remote: dict, payload: dict) -> bool:
    """严格核对远端任务详情是否包含本次提交的核心评分数据。

    远端详情是评分表的投影,不保留本地 task 明细、耗时、seed 等附加字段。
    因此只比较能够唯一定位提交并决定排名的身份字段、总分和环境汇总。
    """
    if not isinstance(remote, dict) or remote.get("task_id") != task_id:
        return False

    result = remote.get("result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return False
    if not isinstance(result, dict):
        return False
    # 旧 backend 的任务详情不保存 benchmark。缺失时不能仅凭这一点否定
    # 已落库结果，但只要远端明确返回了 benchmark，就必须与本地一致。
    remote_benchmark = result.get("benchmark")
    if remote_benchmark is not None and remote_benchmark != payload.get("benchmark"):
        return False

    if remote.get("status") not in ("done", "scored", "failed"):
        return False

    remote_hotkey = remote.get("miner_hotkey") or remote.get("hotkey")
    identities = (
        (remote_hotkey, payload.get("miner_hotkey")),
        (remote.get("hf_repo_id"), payload.get("hf_repo_id")),
        (remote.get("hf_commit"), payload.get("hf_commit")),
        (remote.get("round_num"), payload.get("round_num")),
    )
    if any(actual is None or expected is None or actual != expected for actual, expected in identities):
        return False

    if not isinstance(result.get("success"), bool) or result["success"] is not payload.get("success"):
        return False
    if not _same_number(result.get("total_score"), payload.get("total_score")):
        return False
    remote_envs = result.get("env_scores")
    expected_envs = payload.get("env_scores")
    if not isinstance(remote_envs, list) or not isinstance(expected_envs, list):
        return False

    def by_name(items: list) -> dict | None:
        mapped = {}
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("env_name"), str):
                return None
            name = item["env_name"]
            if name in mapped:
                return None
            mapped[name] = item
        return mapped

    actual_by_name = by_name(remote_envs)
    expected_by_name = by_name(expected_envs)
    if actual_by_name is None or expected_by_name is None or actual_by_name.keys() != expected_by_name.keys():
        return False
    for name, expected in expected_by_name.items():
        actual = actual_by_name[name]
        if actual.get("base_suite") != expected.get("base_suite"):
            return False
        if actual.get("perturbation") != expected.get("perturbation"):
            return False
        if not _same_number(actual.get("score"), expected.get("score")):
            return False
        actual_samples = actual.get("samples")
        expected_samples = expected.get("samples")
        if (
            isinstance(actual_samples, bool)
            or isinstance(expected_samples, bool)
            or not isinstance(actual_samples, int)
            or not isinstance(expected_samples, int)
            or actual_samples != expected_samples
        ):
            return False
    return True


def reconcile_persisted_score(
    client: BackendClient,
    store: StateStore,
    task_id: str,
    payload: dict,
    reason: str,
) -> bool:
    """读取远端结果消解 POST 超时/5xx;完全匹配时把本地账本置为已提交。"""
    try:
        remote = client.fetch_submission(task_id)
    except BackendError as e:
        logger.info(f"{task_id} ({_model_label(payload)}): could not verify remote score after {reason} ({e})")
        return False
    if not remote_score_matches(task_id, remote, payload):
        logger.warning(
            f"{task_id} ({_model_label(payload)}): remote score does not exactly match local payload after {reason}"
        )
        return False
    store.update(
        task_id,
        status="submitted",
        submit_attempts=0,
        submit_error=None,
        next_submit_at=None,
        note=f"remote score verified after {reason}",
    )
    logger.info(
        f"{task_id} ({_model_label(payload)}): remote score verified after {reason}; submission already persisted"
    )
    return True


def try_submit(client: BackendClient, store: StateStore, task_id: str, payload: dict) -> bool:
    """提交一次评分。返回 True 表示已到终态(成功或永久拒绝),False 表示可重试。"""
    payload = prepare_submit_payload(payload)
    label = _model_label(payload)
    # 记录真正交给 HTTP client 的请求体，而不是含本地 task 明细的原始 payload。
    # 鉴权信息位于 X-API-Key header，不在 body 中，因此这里不会泄露 API key。
    logger.info(
        "POST /api/v1/benchmark/task/%s/score payload:\n%s",
        task_id,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
    )
    try:
        client.submit_score(task_id, payload)
    except BackendError as e:
        # 只有连接中断/超时和 5xx 的结果是不确定的。明确 4xx 表示请求未被
        # 接受,绝不能被一个服务端脏 result 误判成提交成功。
        ambiguous = e.status is None or e.status >= 500
        if ambiguous and reconcile_persisted_score(
            client, store, task_id, payload, reason=f"ambiguous POST error: {e}"
        ):
            return True
        if e.permanent:
            logger.error(f"{task_id} ({label}): backend permanently rejected score ({e}); marking abandoned")
            store.update(task_id, status="abandoned", submit_error=str(e))
            return True
        if e.status in (401, 403):
            delay, next_at = _schedule_submit_retry(store, task_id, str(e))
            logger.error(
                f"{task_id} ({label}): backend rejected our API key ({e}); "
                f"check --admin-api-key / BACKEND_ADMIN_API_KEY matches the backend admin_key; "
                f"retry in {delay}s (at {next_at})"
            )
            return False
        delay, next_at = _schedule_submit_retry(store, task_id, str(e))
        logger.warning(f"{task_id} ({label}): score submission failed ({e}); retry in {delay}s (at {next_at})")
        return False
    store.update(
        task_id,
        status="submitted",
        submit_attempts=0,
        submit_error=None,
        next_submit_at=None,
    )
    logger.info(
        f"{task_id} ({label}): score submitted (success={payload['success']} total_score={payload['total_score']})"
    )
    return True


def select_init_seed(task: dict) -> int | None:
    """该任务 init states 随机化的基准 seed = 队列条目自带的 seed 字段。

    seed 由后端入队时按 sha256(block_hash:round_num:drand_random) 派生
    (prototype 仓库 docs/SEED_GENERATION.md):miner 提交权重前不可预知,
    评测后可独立验证,由它派生的 init states(init_mix.derive_task_seed)
    因此同样可复现。缺失或非法时回退纯官方 init states(返回 None),
    不本地造随机数——私自选的 seed 无法向 miner 证明来源。
    """
    raw = task.get("seed")
    if raw is None:
        logger.warning(f"task {task.get('task_id')} has no seed field; using official init states only")
        return None
    # 只认 int 和十进制字符串:float 会被 int() 静默截断,bool 是 int 子类,
    # 都按非法处理——seed 必须逐位准确,宁可退回纯官方也不评一个错的种子。
    seed = -1
    if isinstance(raw, int) and not isinstance(raw, bool):
        seed = raw
    elif isinstance(raw, str):
        try:
            seed = int(raw, 10)
        except ValueError:
            pass
    if not 0 <= seed < 2**32:
        logger.warning(f"task {task.get('task_id')} has invalid seed {raw!r}; using official init states only")
        return None
    return seed


###############################
## Core Loop
###############################
def worker_loop(local_q: queue.Queue, client: BackendClient, store: StateStore, args) -> None:
    while not stop_event.is_set():
        try:
            item = local_q.get(timeout=2)
        except queue.Empty:
            continue
        if item is None:  # --once 模式的结束哨兵
            return
        task_id = item["task_id"]
        with _inflight_lock:
            _inflight.add(task_id)
        try:
            logger.info(f"processing task {task_id} ({_model_label(item)})")
            process_task(item, client, store, args)
            entry = store.get(task_id)
            if entry is not None and entry.get("status") == "pending" and not stop_event.is_set():
                local_q.put(item)  # 暂时性失败(如下载重试耗尽):排到队尾稍后再试
        except Exception:
            logger.exception(f"{task_id} ({_model_label(item)}): unexpected worker error")
            entry = store.get(task_id)
            if entry is not None and entry.get("status") == "done_pending_submit":
                # 提交阶段即使冒出未分类异常也不能丢掉已完成的 payload，
                # 更不能把任务改回 pending 导致整轮昂贵评测重跑。
                store.update(task_id, note="worker crashed during submission; stored score will be retried")
            else:
                store.update(task_id, status="pending", note="worker crashed; will retry on next start")
        finally:
            with _inflight_lock:
                _inflight.discard(task_id)


def process_task(task: dict, client: BackendClient, store: StateStore, args) -> None:
    task_id = task["task_id"]
    t0 = time.time()
    out_dir = args.output_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{task_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_benchmark = getattr(args, "benchmark", "libero")
    store.update(
        task_id,
        status="running",
        task=task,
        benchmark=selected_benchmark,
        protocol_revision=_protocol_revision(selected_benchmark),
        out_dir=str(out_dir),
        # 同一个 task_id 重新入队评测时，旧结果的提交退避不属于这次新结果。
        submit_attempts=0,
        submit_error=None,
        next_submit_at=None,
        note=None,
    )

    # init states 随机化种子 = 队列条目自带的 seed(公开可验证,见 select_init_seed)。
    init_seed = None
    if selected_benchmark != "libero_plus" and not args.no_init_randomization:
        init_seed = select_init_seed(task)

    summary, error = None, ""
    try:
        ref, revision, model_dir = resolve_model(task, args.download_dir, args.allow_local_model)
        if ref:
            logger.info(f"Downloading model {ref} at {revision or 'main'} to {model_dir}")
            download_with_retry(ref, revision, model_dir, args)
        logger.info(f"Evaluating model {ref} at {revision}")
        summary, error = run_evaluation(task, model_dir, out_dir, args, init_seed=init_seed)
    except EvalInterrupted:
        store.update(task_id, status="pending", note="interrupted; will re-run on next start")
        logger.warning(f"{task_id} ({_model_label(task)}): evaluation interrupted; task requeued")
        return
    except DownloadError as e:
        store.update(task_id, status="pending", note=f"download failed after retries: {e}")
        logger.error(f"{task_id} ({_model_label(task)}): download failed after retries; task requeued, no report sent")
        return
    except EvalInfrastructureError as e:
        store.update(task_id, status="pending", note=f"evaluation infrastructure unavailable: {e}")
        logger.error(
            f"{task_id} ({_model_label(task)}): evaluation infrastructure unavailable; task requeued, no report sent"
        )
        return
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.error(f"{task_id} ({_model_label(task)}): evaluation failed: {error}")

    payload = build_score_payload(
        task,
        summary,
        time.time() - t0,
        error,
        init_seed=init_seed,
        benchmark=selected_benchmark,
    )
    (out_dir / "score_payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    store.update(task_id, status="done_pending_submit", payload=payload)

    if not try_submit(client, store, task_id, payload):
        logger.warning(f"{task_id} ({_model_label(task)}): completed score retained; ledger backoff controls retry")


# ----------------------------------------------------------------------------
# 主循环:轮询 + 去重 + 补交
# ----------------------------------------------------------------------------
def classify_queued_task(entry: dict | None, task: dict, benchmark: str | None = None) -> str:
    """决定后端队列任务的处置(纯函数,供单测)。

    返回:
      "evaluate"  没见过的任务,或同 task_id 换了 repo/commit(miner 重新
                  提交)—— 入队评测。换 commit 时若旧结果尚未提交,直接作废:
                  后端只关心当前提交。
      "resubmit"  同一提交已评完、后端也确认收过分,但队列里又出现了 ——
                  后端可能丢了结果,重发已存 payload。对面在等,不能沉默。
      "skip"      本地正在排队/评测(等它出结果),或后端已永久拒绝(abandoned)。
    """
    if entry is None:
        return "evaluate"
    if entry.get("status") in ("pending", "running"):
        return "skip"
    prev = entry.get("task") or {}
    if benchmark is not None and entry.get("benchmark") != benchmark:
        return "evaluate"
    expected_revision = _protocol_revision(benchmark)
    if expected_revision is not None and entry.get("protocol_revision") != expected_revision:
        return "evaluate"
    if (task.get("hf_repo_id"), task.get("hf_commit")) != (prev.get("hf_repo_id"), prev.get("hf_commit")):
        return "evaluate"
    if entry.get("status") == "submitted" and entry.get("payload"):
        return "resubmit"
    return "skip"  # done_pending_submit 走补交路径;abandoned 不再纠缠


def _handle_signal(_signum, _frame):
    # Python signal handlers会打断主线程任意一条字节码。此处若调用 logging,
    # 恰好中断同一 handler 的 emit 时会触发 "reentrant call";只设置标志,
    # 退出路径会统一记录 benchmark worker stopped。
    stop_event.set()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend-url", required=True)
    p.add_argument(
        "--public-api-key",
        default=os.environ.get("BACKEND_PUBLIC_API_KEY", ""),
        help="读取任务队列/任务详情的 public key(默认 BACKEND_PUBLIC_API_KEY)",
    )
    p.add_argument(
        "--admin-api-key",
        default=os.environ.get("BACKEND_ADMIN_API_KEY", ""),
        help="提交评分的 admin key(默认 BACKEND_ADMIN_API_KEY)",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("BACKEND_API_KEY", ""),
        help="兼容旧部署:同一个 key 同时用于读写(优先级低于两个新参数)",
    )
    p.add_argument(
        "--queue-path",
        default="/api/v1/benchmark/queue",
        help="任务队列路径(默认 /api/v1/benchmark/queue)",
    )
    p.add_argument("--poll-interval", type=float, default=60.0, help="quiry interval")

    p.add_argument("--once", action="store_true", help="只轮询一次,处理完即退出(调试用)")
    p.add_argument(
        "--state-file",
        default=str(VALIDATOR_ROOT / "benchmark_worker_state.json"),
        help="本地任务状态账本(去重与断点恢复的依据)",
    )
    p.add_argument("--output-root", default=str(VALIDATOR_ROOT / "eval_runs"), help="评测产物根目录")
    p.add_argument("--download-dir", default=str(VALIDATOR_ROOT / "hf_models"), help="HF 模型下载目录")
    p.add_argument(
        "--download-strategies", default="hfd", help="模型下载策略顺序(逗号分隔):hfd-mirror,hfd,hub-mirror,hub"
    )

    ## Params below will be passed thourgh to run_eval.py
    p.add_argument("--num-trials", type=int, required=True)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--workers-per-gpu", type=int, default=3)
    p.add_argument(
        "--init-workers-per-gpu",
        type=int,
        default=4,
        help="每张 GPU 并发的 init-state 生成进程数(默认 4;与评测 worker 数独立)",
    )

    p.add_argument(
        "--server-impl",
        choices=("upstream", "batched"),
        default="upstream",
        help="policy server 实现(见 run_eval.py)。batched 配 --workers-per-gpu 6 再快 ~13%%,"
        "但推理路径较新,生产默认保持 upstream",
    )
    p.add_argument("--benchmark", choices=sorted(PROFILES), required=True)
    p.add_argument("--task-ids", default="", help="调试用:只评测这些 task id(逗号分隔)")
    p.add_argument(
        "--no-init-randomization",
        action="store_true",
        help="关闭 init states 混合随机化,退回全官方固定初始状态(仅调试/复现旧行为;"
        "默认用队列条目自带的 seed(公开可验证),一半 trial 用官方 init、一半用重采样 init,"
        "抗评测集过拟合)",
    )
    p.add_argument(
        "--model-architectures",
        default="pi0.5",
        metavar="ARCH[,ARCH...]",
        help="允许评测的模型架构(默认 pi0.5;调试可传 pi0,pi0.5)",
    )
    p.add_argument(
        "--eval-config",
        default=None,
        help="显式 openpi 训练配置名(默认根据 checkpoint 架构自动选择)",
    )
    p.add_argument("--eval-timeout", type=float, default=8 * 3600, help="单次评测超时(秒)")
    p.add_argument(
        "--submit-retries", type=int, default=1, help="兼容旧命令;提交失败现在由状态账本做持久化指数退避,不再原地连发"
    )
    p.add_argument(
        "--download-retries",
        type=int,
        default=4,
        help="模型下载的原地重试次数(指数退避,上限 5 分钟;耗尽后任务重新排队而非上报失败)",
    )
    p.add_argument("--allow-local-model", action="store_true", help="允许 hf_repo_id 为本地路径(仅测试用,生产不要开)")
    args = p.parse_args()
    args.output_root = pathlib.Path(args.output_root).expanduser().resolve()
    args.download_dir = pathlib.Path(args.download_dir).expanduser().resolve()
    args.download_strategies = parse_strategies(args.download_strategies)  # 启动即 fail fast
    if args.api_key:
        args.public_api_key = args.public_api_key or args.api_key
        args.admin_api_key = args.admin_api_key or args.api_key
    if not args.public_api_key:
        p.error("--public-api-key or BACKEND_PUBLIC_API_KEY is required")
    if not args.admin_api_key:
        p.error("--admin-api-key or BACKEND_ADMIN_API_KEY is required")
    if args.benchmark == "libero_plus":
        if args.num_trials != 1:
            p.error("libero_plus official protocol requires --num-trials 1")
        if args.task_ids:
            p.error("benchmark_worker cannot score a LIBERO-Plus --task-ids subset; use run_eval.py for development")
        # Perturbation-specific init states are part of each registered variant.
        args.no_init_randomization = True
    return args


def main():
    _setup_logger()
    args = parse_args()

    # 预检评测环境,避免每个任务都失败后才发现 setup.sh 没跑过。
    from libero_eval.paths import CLIENT_VENV_PY, SERVER_VENV_PY  # 只含路径常量

    for py, what in [(SERVER_VENV_PY, "openpi server venv"), (CLIENT_VENV_PY, "LIBERO client venv")]:
        if not py.exists():
            sys.exit(f"[benchmark_worker] Missing {py} — install the {what} first (bash setup.sh; see README).")

    client = BackendClient(
        args.backend_url,
        public_api_key=args.public_api_key,
        admin_api_key=args.admin_api_key,
        queue_path=args.queue_path,
    )
    store = StateStore(pathlib.Path(args.state_file))
    local_q: queue.Queue = queue.Queue()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # 启动恢复:上次死在排队/评测中的任务重新入队;评完未确认的走补交路径。
    for task_id, entry in store.all_tasks().items():
        if entry.get("status") in ("pending", "running") and entry.get("task"):
            logger.info(f"recover: requeue {task_id} ({_model_label(entry['task'])}, was {entry['status']})")
            store.update(task_id, status="pending")
            local_q.put(entry["task"])

    worker = threading.Thread(target=worker_loop, args=(local_q, client, store, args), daemon=True)
    worker.start()
    logger.info(f"benchmark worker up: backend={args.backend_url} poll={args.poll_interval:.0f}s once={args.once}")

    first_poll = True
    while not stop_event.is_set():
        # 先补交欠账(worker 已放弃原地重试的、或重启前遗留的)。
        # _inflight 保证不会和 worker 线程对同一任务双重提交。
        with _inflight_lock:
            inflight = set(_inflight)
        for task_id, entry in store.all_tasks().items():
            if entry.get("status") == "done_pending_submit" and task_id not in inflight and submit_retry_due(entry):
                expected_revision = _protocol_revision(args.benchmark)
                if entry.get("benchmark") != args.benchmark or (
                    expected_revision is not None and entry.get("protocol_revision") != expected_revision
                ):
                    store.update(
                        task_id,
                        status="stale",
                        note="benchmark profile or protocol changed; old score will not be submitted",
                    )
                    continue
                if reconcile_persisted_score(
                    client,
                    store,
                    task_id,
                    prepare_submit_payload(entry["payload"]),
                    reason="scheduled retry",
                ):
                    continue
                try_submit(client, store, task_id, entry["payload"])

        try:
            tasks = client.fetch_queue()
        except BackendError as e:
            logger.warning(f"queue poll failed: {e}")
            tasks = None
        new = 0
        for t in tasks or []:
            tid = t.get("task_id")
            if not tid:
                continue
            with _inflight_lock:
                if tid in _inflight:
                    continue  # worker 线程正处理中,下一轮再看
            entry = store.get(tid)
            verdict = classify_queued_task(entry, t, args.benchmark)
            if verdict == "evaluate":
                if entry is not None:
                    prev = entry.get("task") or {}
                    logger.info(
                        f"{tid}: backend re-queued with new submission "
                        f"{prev.get('hf_repo_id')}@{str(prev.get('hf_commit'))[:12]} -> "
                        f"{t.get('hf_repo_id')}@{str(t.get('hf_commit'))[:12]}; re-evaluating"
                    )
                store.update(tid, status="pending", task=t, benchmark=args.benchmark, payload=None)
                local_q.put(t)
                new += 1
            elif verdict == "resubmit" and entry is not None:
                # 同一提交的结果早已交过,但后端仍把任务挂在队列里 ——
                # 重发已存结果,保证对面等得到回音,且不重复评测。
                if not submit_retry_due(entry):
                    continue
                logger.info(
                    f"{tid} ({_model_label(t)}): already evaluated (result on file); re-submitting stored score"
                )
                try_submit(client, store, tid, entry["payload"])
        if new:
            logger.info(f"picked up {new} new task(s) from backend queue")
        elif first_poll and tasks is not None:
            # 首轮空转也出一条日志,否则启动后长时间静默像卡死。
            if tasks:
                logger.info(
                    f"queue has {len(tasks)} task(s), all already evaluated in the local "
                    f"ledger ({args.state_file}); stored results were re-sent where applicable"
                )
            else:
                logger.info(f"queue empty; polling every {args.poll_interval:.0f}s")
        if tasks is not None:
            first_poll = False

        if args.once:
            local_q.put(None)
            break
        stop_event.wait(args.poll_interval)

    worker.join()
    logger.info("benchmark worker stopped")


if __name__ == "__main__":
    main()
