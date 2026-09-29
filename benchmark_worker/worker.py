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
import copy
import datetime
import json
import logging
import math
import os
import pathlib
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

VALIDATOR_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VALIDATOR_ROOT))

from benchmark_worker.backend_client import BackendClient, BackendError, TaskInvalidatedError
from benchmark_worker.axis_rotation import AxisRotation, default_rotation_directory
from benchmark_worker.axis_randomization import queue_seed, verify_summary as verify_axis_randomized_summary
from benchmark_worker.profiles import (
    BenchmarkNotReadyError,
    configure_axis_profiles,
    get_profile,
    is_axis_benchmark,
    refresh_axis_profiles,
)
from libero_eval.axis_runtime import AXIS_V1_NAME
from benchmark_worker.scoring import (
    build_score_payload,
    prepare_submit_payload,
    successful_payload_incomplete_reason,
)
from benchmark_worker.state import StateStore
from libero_eval.download import COMMIT_HASH_RE, DownloadError, DownloadInterrupted, download_model, parse_strategies
from libero_eval.download import MODEL_MAX_BYTES, ModelSizeExceeded, check_local_model_size
from libero_eval.backbones import parse_backbone
from libero_eval.gpu_health import GPU_CHECK_INTERVAL, check_gpu_health

RUN_EVAL = VALIDATOR_ROOT / "libero_eval" / "run_eval.py"

_REPO_ID_RE = re.compile(r"^[\w][\w.-]*/[\w][\w.-]*$")
_SUITE_RE = re.compile(r"^[a-z0-9_]+$")
BASE_MODELS = ("pi0.5", "lingbot-vla-2.0")
# Editors occasionally leave backup files in a submitted HF repo. They are not
# model inputs; accepting their absence is safe only because download_model
# still verifies every non-matching artifact against the pinned commit.
MODEL_OPTIONAL_ARTIFACT_PATTERNS = ["*.bak"]

stop_event = threading.Event()
_shutdown_requested = False
_poll_wakeup_event = threading.Event()
_inflight_lock = threading.Lock()
_inflight: set[str] = set()  # task_id 正被 worker 线程处理(评测或提交中)

_BENCHMARK_PROTOCOL_REVISIONS = {
    "libero_plus": "libero_plus_official_v1",
    "robotwin": "robotwin_lingbot_v2_clean_official_v1",
}

logger = logging.getLogger("benchmark_worker")


def _stopping() -> bool:
    return _shutdown_requested or stop_event.is_set()


def _wait_for_event(event: threading.Event, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not _stopping():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return event.is_set()
        if event.wait(min(remaining, 0.2)):
            return True
    return False


class _StderrLogWriter:
    """Route process stderr lines through the configured worker logger.

    The logger's console handler keeps writing to the original stderr stream,
    while its file handler persists the same line. This captures uncaught
    tracebacks, warnings, and direct ``print(..., file=sys.stderr)`` output
    without changing the subprocess-specific run_eval.log redirection.
    """

    def __init__(self, target_logger: logging.Logger, original_stream):
        self.target_logger = target_logger
        self.original_stream = original_stream
        self._buffer = ""
        self._lock = threading.RLock()

    @property
    def encoding(self):
        return getattr(self.original_stream, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self.original_stream, "errors", "replace")

    def isatty(self) -> bool:
        return bool(getattr(self.original_stream, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.original_stream.fileno()

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not text:
            return 0
        if not isinstance(text, str):
            text = str(text)
        with self._lock:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._emit_line(line)
        return len(text)

    def flush(self) -> None:
        with self._lock:
            if self._buffer:
                self._emit_line(self._buffer)
                self._buffer = ""
            for handler in self.target_logger.handlers:
                handler.flush()

    def _emit_line(self, line: str) -> None:
        line = line.rstrip("\r")
        if line:
            self.target_logger.error("stderr: %s", line)


class _DatedDailyFileHandler(logging.FileHandler):
    """Write directly to a dated file and switch files at local midnight."""

    def __init__(self, log_dir: pathlib.Path, retention_days: int = 30, date_provider=None):
        if retention_days < 1:
            raise ValueError("retention_days must be positive")
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._date_provider = date_provider or datetime.date.today
        self.current_date = self._date_provider()
        super().__init__(self._path_for(self.current_date), encoding="utf-8")
        self._delete_expired_files()

    def _path_for(self, day: datetime.date) -> pathlib.Path:
        return self.log_dir / f"benchmark_worker-{day.isoformat()}.log"

    @property
    def current_log_path(self) -> pathlib.Path:
        return self._path_for(self.current_date)

    def _delete_expired_files(self) -> None:
        oldest_kept = self.current_date - datetime.timedelta(days=self.retention_days - 1)
        for path in self.log_dir.glob("benchmark_worker-????-??-??.log"):
            try:
                file_date = datetime.date.fromisoformat(path.stem.removeprefix("benchmark_worker-"))
            except ValueError:
                continue
            if file_date < oldest_kept:
                try:
                    path.unlink()
                except OSError:
                    pass  # Retention cleanup must not prevent current logs from being written.

    def _switch_date_if_needed(self) -> None:
        today = self._date_provider()
        if today == self.current_date:
            return
        if self.stream is not None:
            self.stream.flush()
            self.stream.close()
        self.current_date = today
        self.baseFilename = os.path.abspath(self._path_for(today))
        self.stream = self._open()
        self._delete_expired_files()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._switch_date_if_needed()
            super().emit(record)
        except Exception:
            self.handleError(record)


def _protocol_revision(benchmark: str | None) -> str | None:
    if is_axis_benchmark(benchmark):
        return get_profile(benchmark).protocol_revision or _BENCHMARK_PROTOCOL_REVISIONS.get(benchmark)
    return _BENCHMARK_PROTOCOL_REVISIONS.get(benchmark or "")


def select_benchmark(task: dict, override: str | None = None) -> str:
    """CLI overrides queue routing; otherwise require a supported queue benchmark."""
    benchmark = override if override is not None else task.get("benchmark")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("queue task must include benchmark when --benchmark is omitted")
    get_profile(benchmark)
    expected_revision = _protocol_revision(benchmark)
    task_revision = task.get("protocol_revision")
    if override is None and task_revision is not None and task_revision != expected_revision:
        raise ValueError(
            f"queue protocol_revision {task_revision!r} does not match {benchmark} ({expected_revision!r})"
        )
    return benchmark


def task_matches_filter(task: dict, args) -> bool:
    """Restrict queue ownership without overriding the backend's benchmark version."""
    if not getattr(args, "axis_only", False):
        return True
    name = task.get("benchmark")
    return is_axis_benchmark(name)


def task_evaluation_args(task: dict, args):
    """Resolve on a copy so mixed queue tasks cannot change each other's defaults."""
    if not task_matches_filter(task, args):
        raise ValueError("--axis-only skips tasks whose queue benchmark is not AXIS")
    resolved = copy.copy(args)
    resolved.benchmark = select_benchmark(task, getattr(args, "benchmark", None))
    _apply_benchmark_options(resolved)
    if get_profile(resolved.benchmark).randomization_manifest_path is not None:
        resolved.axis_randomization_seed = queue_seed(task)
    return resolved


def pending_score_matches_profile(entry: dict, override: str | None) -> bool:
    try:
        benchmark = select_benchmark(entry.get("task") or {}, override)
    except ValueError:
        return False
    revision = _protocol_revision(benchmark)
    return entry.get("benchmark") == benchmark and (revision is None or entry.get("protocol_revision") == revision)


def _model_label(d: dict | None) -> str:
    """任务或评分 payload 里的模型标识,供日志定位:hf_repo_id@commit 前 12 位。

    task_id 是后端生成的不透明串,单看它不知道评的是哪个模型;所有任务级
    日志都应带上这个标签,读日志的人才能对上"哪个 repo 的哪次提交"。
    """
    d = d or {}
    repo = d.get("hf_repo_id") or "?"
    commit = (d.get("hf_commit") or "")[:12] or "?"
    return f"{repo}@{commit}"


def _backend_log_namespace(backend_url: str) -> str:
    """Return a filesystem-safe namespace derived from the backend authority."""
    parsed = urlsplit(backend_url)
    if not parsed.hostname:
        parsed = urlsplit(f"//{backend_url}")
    if not parsed.hostname:
        raise ValueError(f"backend URL has no hostname: {backend_url!r}")

    authority = parsed.hostname.lower()
    if parsed.port is not None:
        authority = f"{authority}_{parsed.port}"
    namespace = re.sub(r"[^a-z0-9._-]+", "_", authority).strip("._-")
    if not namespace:
        raise ValueError(f"backend URL has no usable hostname: {backend_url!r}")
    return namespace


def _setup_logger(backend_url: str, log_root: pathlib.Path | None = None, date_provider=None) -> pathlib.Path:
    """Configure console logging and a backend-isolated dated daily file."""
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # _setup_logger may be called again in the same process (notably by
    # tests). Always bypass our stderr adapter so logger output cannot feed
    # recursively back into itself.
    console_stream = sys.stderr.original_stream if isinstance(sys.stderr, _StderrLogWriter) else sys.stderr
    console = logging.StreamHandler(console_stream)
    console.setFormatter(fmt)
    logger.addHandler(console)

    root = pathlib.Path(log_root) if log_root is not None else VALIDATOR_ROOT / "logs"
    log_dir = root / _backend_log_namespace(backend_url)
    file_handler = _DatedDailyFileHandler(log_dir, retention_days=30, date_provider=date_provider)
    log_path = file_handler.current_log_path
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return log_path


def _install_stderr_logging() -> _StderrLogWriter:
    """Persist subsequent process-level stderr output in the worker log."""
    if isinstance(sys.stderr, _StderrLogWriter):
        return sys.stderr
    redirected = _StderrLogWriter(logger, sys.stderr)
    sys.stderr = redirected
    return redirected


class EvalInterrupted(Exception):
    """收到退出信号,当前评测被中止(任务会重新排队,不上报失败)。"""


class EvalInfrastructureError(Exception):
    """本机评测基础设施暂时不可用,任务应重新排队且不上报失败。"""


def _remove_cache_dir(path: pathlib.Path, root: pathlib.Path, label: str) -> None:
    """只删除 worker 在指定根目录下直接创建的单个缓存目录。

    清缓存属于破坏性操作，因此不接受根目录本身、嵌套路径、符号链接或
    根目录之外的路径。调用方若清理失败会保留 pending 状态，绝不带着
    可能污染的缓存继续评测。
    """
    path = pathlib.Path(path)
    root = pathlib.Path(root).resolve()
    if path.is_symlink():
        raise RuntimeError(f"refusing to delete symlinked {label}: {path}")
    resolved = path.resolve()
    if resolved == root or resolved.parent != root:
        raise RuntimeError(f"refusing to delete {label} outside its cache root {root}: {resolved}")
    if not resolved.exists():
        return
    if not resolved.is_dir():
        raise RuntimeError(f"refusing to delete non-directory {label}: {resolved}")
    shutil.rmtree(resolved)
    logger.info(f"Deleted {label} before clean retry: {resolved}")


def _failed_output_history(entry: dict, output_dir: pathlib.Path | str | None) -> list[str]:
    """Return the append-only list of failed attempt directories for a task."""
    history = entry.get("failed_out_dirs")
    history = [path for path in history if isinstance(path, str) and path] if isinstance(history, list) else []
    legacy = entry.get("last_failed_out_dir")
    if isinstance(legacy, str) and legacy and legacy not in history:
        history.append(legacy)
    if output_dir is not None:
        current = str(output_dir)
        if current not in history:
            history.append(current)
    return history


def _preserve_failure_artifacts(
    task_id: str,
    args,
    reason: str,
    output_dir: pathlib.Path,
    *,
    download_cache: pathlib.Path | None = None,
) -> list[str]:
    """Persist failure metadata and download logs without modifying arbitrary paths."""
    output_dir = pathlib.Path(output_dir)
    output_root = pathlib.Path(args.output_root).resolve()
    resolved = output_dir.resolve()
    if output_dir.is_symlink() or resolved.parent != output_root:
        return [f"refusing to write failure metadata outside output root {output_root}: {resolved}"]
    if not resolved.is_dir():
        return [f"failed output directory is unavailable: {resolved}"]

    errors = []
    failure_path = resolved / "failure.json"
    if not failure_path.exists():
        failure = {
            "task_id": task_id,
            "failed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "reason": reason,
        }
        try:
            failure_path.write_text(json.dumps(failure, indent=2, ensure_ascii=False))
        except OSError as e:
            errors.append(f"could not write {failure_path}: {e}")

    if download_cache is not None:
        download_log = pathlib.Path(download_cache) / ".hfd" / "console.log"
        if download_log.is_file():
            try:
                shutil.copy2(download_log, resolved / "download.log")
            except OSError as e:
                errors.append(f"could not preserve {download_log}: {e}")
    return errors


def _schedule_clean_retry(
    store: StateStore,
    task_id: str,
    args,
    reason: str,
    *,
    download_cache: pathlib.Path | None = None,
    evaluation_cache: pathlib.Path | None = None,
) -> None:
    """清理不可复用的下载缓存，保留评测诊断产物并重新排队。"""
    previous = store.get(task_id) or {}
    artifact_errors = []
    failed_out_dir = str(evaluation_cache) if evaluation_cache is not None else None
    if evaluation_cache is not None:
        artifact_errors = _preserve_failure_artifacts(
            task_id,
            args,
            reason,
            pathlib.Path(evaluation_cache),
            download_cache=download_cache,
        )

    cleanup_errors = []
    pending_download = None
    if download_cache is not None:
        try:
            _remove_cache_dir(download_cache, args.download_dir, "download cache")
        except Exception as e:
            cleanup_errors.append(f"download cache: {e}")
            pending_download = str(download_cache)

    preserved = f"; evaluation artifacts preserved at {failed_out_dir}" if failed_out_dir else ""
    note = f"{reason}{preserved}; queued for a full retry"
    if artifact_errors:
        note += f"; artifact preservation warning ({'; '.join(artifact_errors)})"
    if cleanup_errors:
        note += f"; cache cleanup pending ({'; '.join(cleanup_errors)}); retry is blocked until clean"
    store.update(
        task_id,
        status="pending",
        payload=None,
        out_dir=None,
        last_failed_out_dir=failed_out_dir or previous.get("last_failed_out_dir"),
        failed_out_dirs=_failed_output_history(previous, evaluation_cache),
        resume_evaluation=False,
        cleanup_download_dir=pending_download,
        # Old versions used this field to delete failed evaluation directories.
        # Keep it cleared: output directories are diagnostic artifacts, not caches.
        cleanup_evaluation_dir=None,
        note=note,
    )


def _finish_pending_cleanup(task_id: str, store: StateStore, args, previous: dict) -> bool:
    """清理上次遗留的下载缓存，并迁移旧账本中的评测产物引用。"""
    download_cache = previous.get("cleanup_download_dir")
    evaluation_cache = previous.get("cleanup_evaluation_dir")
    # 兼容旧版本或进程崩溃留下的部分输出。整轮仍然从头重跑，但旧目录
    # 只作为诊断产物保留，不参与 resume，也绝不在自动重试中删除。
    if previous.get("out_dir") and (
        previous.get("status") in ("pending", "running") or previous.get("resume_evaluation")
    ):
        evaluation_cache = previous["out_dir"]
    if not download_cache and not evaluation_cache:
        return True

    _schedule_clean_retry(
        store,
        task_id,
        args,
        "finishing cleanup from the previous failed attempt",
        download_cache=pathlib.Path(download_cache) if download_cache else None,
        evaluation_cache=pathlib.Path(evaluation_cache) if evaluation_cache else None,
    )
    entry = store.get(task_id) or {}
    return not entry.get("cleanup_download_dir")


def _create_output_dir(output_root: pathlib.Path, task_id: str) -> pathlib.Path:
    """Atomically allocate a unique directory for one evaluation attempt."""
    output_root = pathlib.Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("._-") or "task"
    base = output_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_task_id}"
    for attempt in range(1, 10_000):
        candidate = base if attempt == 1 else base.with_name(f"{base.name}_attempt{attempt}")
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"could not allocate a unique evaluation output directory below {output_root}")


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


def select_base_model(task: dict, benchmark: str | None = None) -> str:
    """Validate and return the queue-owned base-model selection.

    The backend queue is the production source of truth for model identity.
    Keep its wire values stable and pass them directly to ``run_eval.py``;
    that CLI owns alias normalization for the evaluator runtime.
    """
    raw = task.get("base_model")
    base_model = raw if isinstance(raw, str) else ""
    if base_model not in BASE_MODELS:
        raise ValueError(f"invalid base_model {raw!r}: queue task must provide one of {', '.join(BASE_MODELS)}")
    if benchmark == "robotwin" and base_model != "lingbot-vla-2.0":
        raise ValueError("robotwin tasks require base_model 'lingbot-vla-2.0'")
    if is_axis_benchmark(benchmark) and base_model != "pi0.5":
        raise ValueError(f"{benchmark} tasks require base_model 'pi0.5'")
    return base_model


def resolve_evaluator_source_commit(explicit: str | None = None) -> str:
    """Resolve an auditable evaluator revision for production score artifacts.

    An explicit revision is used by immutable source bundles without ``.git``.
    A normal checkout is accepted only when tracked and untracked source state
    is clean, otherwise the HEAD hash would misrepresent the code that scored
    a model. Documentation and archived deployment evidence are not imported
    by the evaluator and may contain local reports without blocking startup.
    """
    if explicit:
        if not COMMIT_HASH_RE.fullmatch(explicit):
            raise ValueError("evaluator source commit must be a full 40-character lowercase Git hash")
        return explicit
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=VALIDATOR_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--",
                ".",
                ":(exclude)docs",
                ":(exclude)deploy/evidence",
            ],
            cwd=VALIDATOR_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "cannot determine evaluator source revision; deploy a clean Git checkout or pass "
            "--evaluator-source-git-commit for an immutable source bundle"
        ) from exc
    if not COMMIT_HASH_RE.fullmatch(head):
        raise ValueError(f"git rev-parse returned an invalid evaluator source commit: {head!r}")
    if status:
        paths = [line[3:] if len(line) > 3 else line for line in status.splitlines()]
        preview = ", ".join(paths[:5])
        suffix = " ..." if len(paths) > 5 else ""
        raise ValueError(
            f"evaluator checkout has uncommitted source files ({preview}{suffix}); "
            "refusing to publish scores with a false revision"
        )
    return head


def download_with_retry(
    ref: str, revision: str | None, model_dir: pathlib.Path, args, *, max_total_bytes: int | None = None
) -> None:
    """下载模型,对失败做有上限的原地重试。

    网络/镜像故障重试若干次,耗尽后由调用方清缓存并重新排队。仓库缺文件、
    官方完整性校验失败等 permanent 错误不在本轮原地重试，但同样不生成
    评测结论；调用方清理整个下载目录后把任务放回队尾。
    """
    for attempt in range(args.download_retries):
        if _stopping():
            raise EvalInterrupted()
        try:
            download_model(
                ref,
                model_dir,
                revision=revision,
                strategies=args.download_strategies,
                optional_patterns=MODEL_OPTIONAL_ARTIFACT_PATTERNS,
                max_total_bytes=max_total_bytes,
                log=logger.info,
            )
            if _stopping():
                raise EvalInterrupted()
            return
        except DownloadInterrupted as exc:
            raise EvalInterrupted() from exc
        except DownloadError as e:
            if _stopping():
                raise EvalInterrupted() from e
            if e.permanent:
                raise
            if attempt + 1 >= args.download_retries:
                raise
            wait = min(300, 30 * 2**attempt)
            logger.warning(f"download failed (attempt {attempt + 1}/{args.download_retries}): {e}; retrying in {wait}s")
            if _wait_for_event(stop_event, wait) or _stopping():
                raise EvalInterrupted() from None


# ----------------------------------------------------------------------------
# 评测子进程
# ----------------------------------------------------------------------------
def _terminate(proc: subprocess.Popen) -> None:
    # 先礼后兵:SIGTERM 让 run_eval 自己清理 policy server,超时再杀整个会话
    # (start_new_session=True 保证 pid 即 pgid)。
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error("evaluation pid=%s did not exit after SIGKILL; GPU driver may be blocked", proc.pid)


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


def _report_progress(client: BackendClient, task_id: str, stage: str, detail: dict, worker_id: str) -> bool:
    """Best-effort progress reporting, except explicit task invalidation conflicts."""
    try:
        client.report_progress(task_id, stage, detail, worker_id)
    except TaskInvalidatedError:
        # 这不是可忽略的可观测性故障：后端已明确要求停止当前任务。
        raise
    except BackendError as e:
        logger.warning(f"{task_id}: progress update failed (stage={stage} worker={worker_id}): {e}")
        return False
    logger.info(f"{task_id}: progress stage={stage} detail={detail} worker={worker_id}")
    return True


def _forward_progress_events(
    progress_path: pathlib.Path,
    offset: int,
    callback,
) -> int:
    """Forward complete JSONL events written by run_eval.py and return the new byte offset."""
    try:
        with progress_path.open("r", encoding="utf-8") as progress_file:
            progress_file.seek(offset)
            while True:
                line_start = progress_file.tell()
                line = progress_file.readline()
                if not line:
                    return progress_file.tell()
                if not line.endswith("\n"):
                    return line_start  # writer has not completed this event yet
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Ignoring malformed progress event in {progress_path}: {e}")
                    continue
                if not isinstance(event, dict) or event.get("stage") != "evaluating":
                    logger.warning(f"Ignoring invalid progress event in {progress_path}: {event!r}")
                    continue
                detail = event.get("detail")
                if not isinstance(detail, dict):
                    logger.warning(f"Ignoring progress event without detail object in {progress_path}: {event!r}")
                    continue
                callback("evaluating", detail)
    except FileNotFoundError:
        return offset


def run_evaluation(
    task: dict,
    model_dir: pathlib.Path,
    out_dir: pathlib.Path,
    args,
    init_seed: int | None = None,
    progress_callback=None,
):
    """运行一次 run_eval.py,返回 (summary | None, error_str)。

    init_seed 非空时启用 init states 混合随机化(run_eval --init-seed:
    每 task 一半 trial 用官方初始状态、一半用该 seed 重采样的初始状态)。
    收到退出信号时中止子进程并抛 EvalInterrupted。
    """
    args = task_evaluation_args(task, args)
    selected_benchmark = args.benchmark
    profile = get_profile(selected_benchmark)
    benchmark = profile.runtime_benchmark
    base_model = select_base_model(task, selected_benchmark)
    server_impl = (
        getattr(args, "lingbot_server_impl", None) or args.server_impl
        if base_model == "lingbot-vla-2.0"
        else args.server_impl
    )
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
        "--backbone",
        base_model,
        "--num-trials",
        str(args.num_trials),
        "--gpus",
        args.gpus,
        "--workers-per-gpu",
        str(args.workers_per_gpu),
        "--init-workers-per-gpu",
        str(getattr(args, "init_workers_per_gpu", 4)),
        "--server-impl",
        server_impl,
        "--max-batch",
        str(getattr(args, "max_batch", 4)),
        "--output-dir",
        str(out_dir),
    ]
    evaluator_source_commit = getattr(args, "evaluator_source_git_commit", None)
    if evaluator_source_commit:
        cmd += ["--evaluator-source-git-commit", evaluator_source_commit]
    if args.eval_config:
        cmd += ["--config", args.eval_config]
    if profile.runtime_benchmark == "axis":
        cmd += ["--axis-manifest", str(profile.manifest_path)]
    if profile.randomization_manifest_path is not None:
        cmd += [
            "--axis-randomization-manifest",
            str(profile.randomization_manifest_path),
            "--axis-randomization-seed",
            str(args.axis_randomization_seed),
        ]
    if profile.policy_seed is not None:
        cmd += ["--seed", str(profile.policy_seed)]
    if getattr(args, "lingbot_norm_stats", None):
        cmd += ["--lingbot-norm-stats", args.lingbot_norm_stats]
    if args.task_ids:
        cmd += ["--task-ids", args.task_ids]
    # LIBERO-Plus variants ship perturbation-specific init states; replacing
    # them changes the benchmark definition and run_eval correctly rejects it.
    if init_seed is not None and benchmark in ("libero", "libero_pro"):
        cmd += ["--init-seed", str(init_seed)]

    progress_path = out_dir / "progress.jsonl"
    progress_path.write_text("")
    cmd += ["--progress-file", str(progress_path)]

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
        next_gpu_check = time.monotonic() + GPU_CHECK_INTERVAL
        progress_offset = 0
        try:
            while proc.poll() is None:
                if progress_callback is not None:
                    progress_offset = _forward_progress_events(progress_path, progress_offset, progress_callback)
                if _stopping():
                    _terminate(proc)
                    raise EvalInterrupted()
                if time.time() > deadline:
                    _terminate(proc)
                    raise EvalInfrastructureError(f"evaluation timed out after {args.eval_timeout:.0f}s")
                if time.monotonic() >= next_gpu_check:
                    health = check_gpu_health()
                    next_gpu_check = time.monotonic() + GPU_CHECK_INTERVAL
                    if not health.healthy:
                        _terminate(proc)
                        raise EvalInfrastructureError(f"GPU health check failed: {health.detail}")
                time.sleep(5)
            if progress_callback is not None:
                _forward_progress_events(progress_path, progress_offset, progress_callback)
        except TaskInvalidatedError:
            # progress callback 可以收到 SUPERSEDED/REJECTED。异常离开本层前
            # 必须结束整个评测进程组，不能留下继续占用 GPU 的孤儿进程。
            if proc.poll() is None:
                _terminate(proc)
            raise

    # SIGTERM 与子进程自行退出可能发生在同一个轮询间隔内。停机意图必须
    # 优先于刚结束的子进程结果,否则会把中断/OOM 当成最终模型结论提交。
    if _stopping():
        raise EvalInterrupted()

    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        log_text = log_path.read_text(errors="replace")
        tail = "\n".join(log_text.splitlines()[-40:]) or "<run_eval.log is empty>"
        if proc.returncode == 3:
            # 模型合法性检查未通过(libero_eval/check_model.py):把拒绝理由
            # 带回后端,miner 能直接看到原因。
            return None, f"model rejected by pre-eval check:\n{tail}"
        if _has_retryable_infrastructure_error(log_text):
            raise EvalInfrastructureError(f"evaluation infrastructure failed:\n{tail}")
        raise EvalInfrastructureError(
            f"run_eval exited {proc.returncode} without a complete summary (see {log_path}):\n{tail}"
        )
    summary = json.loads(summary_path.read_text())
    if profile.manifest_sha256 is not None:
        expected_metadata = {
            "benchmark": selected_benchmark,
            "protocol_revision": profile.protocol_revision,
            "manifest_canonical_sha256": profile.manifest_sha256,
            "policy_seed": profile.policy_seed,
            "num_trials_per_task": profile.expected_trials_per_task,
            "dry_run": False,
        }
        mismatched = [key for key, value in expected_metadata.items() if summary.get(key) != value]
        if mismatched:
            raise EvalInfrastructureError(
                f"evaluation does not match the worker's benchmark config: {', '.join(mismatched)}; "
                "restore the frozen version or publish a new version"
            )
    if selected_benchmark in ("libero_plus", "robotwin"):
        protocol = summary.get("evaluation_protocol") or {}
        if not protocol.get("official_result"):
            deviations = "; ".join(protocol.get("deviations") or ["missing official protocol metadata"])
            raise EvalInfrastructureError(f"refusing to score non-official {selected_benchmark} result: {deviations}")
    tasks = summary.get("tasks")
    if not isinstance(tasks, dict):
        raise EvalInfrastructureError("evaluation summary has no task results; refusing to score it")
    expected_tasks = profile.expected_task_count
    failed = sorted(n for n, r in tasks.items() if not isinstance(r, dict) or r.get("status") != "ok")
    wrong_trials = sorted(
        n
        for n, r in tasks.items()
        if isinstance(r, dict) and r.get("status") == "ok" and r.get("num_trials") != args.num_trials
    )
    if len(tasks) != expected_tasks or failed or wrong_trials:
        details = []
        if len(tasks) != expected_tasks:
            details.append(f"summary contains {len(tasks)}/{expected_tasks} required tasks")
        if failed:
            details.append(f"{len(failed)} task(s) did not complete: {', '.join(failed[:10])}")
        if wrong_trials:
            details.append(f"{len(wrong_trials)} task(s) have incomplete trial counts: {', '.join(wrong_trials[:10])}")
        infra_detail = _failed_task_infrastructure_detail(out_dir, failed)
        if infra_detail:
            details.append(infra_detail)
        raise EvalInfrastructureError(
            "incomplete evaluation; refusing to publish a partial score:\n" + "\n".join(details)
        )
    if profile.expected_task_ids is not None:
        try:
            actual_task_ids = {int(result["task_id"]) for result in tasks.values() if isinstance(result, dict)}
        except (KeyError, TypeError, ValueError) as exc:
            raise EvalInfrastructureError("evaluation summary contains an invalid task id") from exc
        if actual_task_ids != set(profile.expected_task_ids):
            raise EvalInfrastructureError(
                "evaluation task ids do not match the frozen benchmark manifest; refusing to score"
            )
    if proc.returncode != 0:
        raise EvalInfrastructureError(
            f"run_eval exited {proc.returncode} despite a complete-looking summary; refusing to score it"
        )
    if profile.randomization_manifest_path is not None:
        try:
            verify_axis_randomized_summary(summary, profile, args.axis_randomization_seed)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise EvalInfrastructureError(f"invalid AXIS randomization evidence: {exc}") from exc
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
    expected_benchmark = payload.get("benchmark")
    remote_benchmark = result.get("benchmark")
    if is_axis_benchmark(expected_benchmark) and remote_benchmark != expected_benchmark:
        return False
    if remote_benchmark is not None and remote_benchmark != expected_benchmark:
        return False
    remote_revision = result.get("protocol_revision")
    expected_revision = payload.get("protocol_revision")
    if is_axis_benchmark(expected_benchmark) and remote_revision != expected_revision:
        return False
    if remote_revision is not None and remote_revision != expected_revision:
        return False

    # Production task detail uses ``evaluated`` after a successful score POST;
    # older deployments used ``done``/``scored``. All are terminal read-back
    # states, while pending/running must never reconcile an ambiguous POST.
    if remote.get("status") not in ("evaluated", "done", "scored", "failed"):
        return False

    remote_hotkey = remote.get("miner_hotkey") or remote.get("hotkey")
    identities = (
        (remote_hotkey, payload.get("miner_hotkey")),
        (remote.get("hf_repo_id"), payload.get("hf_repo_id")),
        (remote.get("hf_commit"), payload.get("hf_commit")),
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


def _invalidate_task(store: StateStore, task_id: str, error: TaskInvalidatedError) -> None:
    """把后端已作废的提交置为本地终态，并唤醒主线程立即重拉队列。"""
    status = error.code.lower()
    request = f" request_id={error.request_id}" if error.request_id else ""
    detail = error.detail or str(error)
    store.update(
        task_id,
        status=status,
        payload=None,
        submit_attempts=0,
        submit_error=None,
        next_submit_at=None,
        note=f"backend returned {error.code}:{request} {detail}",
    )
    logger.info(
        f"{task_id}: backend returned {error.code}{request}; "
        "stopping this task without a score and refreshing the queue"
    )
    _poll_wakeup_event.set()


def try_submit(client: BackendClient, store: StateStore, task_id: str, payload: dict) -> bool:
    """提交一次评分。返回 True 表示已到终态(成功、作废或永久拒绝),False 表示可重试。"""
    incomplete_reason = successful_payload_incomplete_reason(payload)
    if incomplete_reason:
        entry = store.get(task_id) or {}
        failed_out_dir = entry.get("out_dir")
        store.update(
            task_id,
            status="pending",
            payload=None,
            out_dir=None,
            last_failed_out_dir=failed_out_dir or entry.get("last_failed_out_dir"),
            failed_out_dirs=_failed_output_history(entry, failed_out_dir),
            resume_evaluation=False,
            cleanup_evaluation_dir=None,
            submit_error=None,
            next_submit_at=None,
            note=(
                f"blocked incomplete success payload: {incomplete_reason}; "
                f"evaluation artifacts preserved at {failed_out_dir}"
            ),
        )
        logger.error(
            f"{task_id} ({_model_label(payload)}): refusing to submit success=true because {incomplete_reason}"
        )
        return False
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
    except TaskInvalidatedError as e:
        _invalidate_task(store, task_id, e)
        return True
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

    seed 由后端入队时按当前协议派生:miner 提交权重前不可预知,评测后可独立
    验证,由它派生的 init states(init_mix.derive_task_seed)因此同样可复现。
    validator 将其视作不透明的 uint32,不依赖后端的具体派生字段。缺失或非法
    时回退纯官方 init states(返回 None),不本地造随机数——私自选的 seed 无法
    向 miner 证明来源。
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
def wait_for_gpu_health() -> bool:
    """Keep pending work intact while the independent monitor sends notifications."""
    paused = False
    while not _stopping():
        health = check_gpu_health()
        if health.healthy:
            if paused:
                logger.info("GPU health restored; resuming task execution")
            return True
        if not paused:
            logger.error("GPU unavailable; task execution paused: %s", health.detail)
            paused = True
        _wait_for_event(stop_event, GPU_CHECK_INTERVAL)
    return False


def worker_loop(local_q: queue.Queue, client: BackendClient, store: StateStore, args) -> None:
    while not _stopping():
        try:
            item = local_q.get(timeout=2)
        except queue.Empty:
            continue
        if item is None:  # --once 模式的结束哨兵
            return
        if not wait_for_gpu_health():
            return
        task_id = item["task_id"]
        with _inflight_lock:
            _inflight.add(task_id)
        try:
            logger.info(f"processing task {task_id} ({_model_label(item)})")
            process_task(item, client, store, args)
        except BenchmarkNotReadyError as exc:
            # A version can become unavailable after enqueueing. Return to backend polling,
            # without API writes or a busy retry loop on the local FIFO.
            store.update(task_id, status="stale", note=str(exc))
            logger.warning("task %s waits for its benchmark: %s", task_id, exc)
        except TaskInvalidatedError as e:
            # 防御性兜底：未来若新增 POST 调用而未在 process_task 内单独捕获，
            # 也不能被下面的通用异常路径改回 pending 后重新评测。
            _invalidate_task(store, task_id, e)
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
            entry = store.get(task_id)
            if entry is not None and entry.get("status") == "pending" and not _stopping():
                # 所有非终态失败统一回到本地 FIFO 队尾；即使 process_task
                # 冒出未预期异常，也不能丢到只能靠重启恢复。
                local_q.put(item)
            with _inflight_lock:
                _inflight.discard(task_id)


def process_task(task: dict, client: BackendClient, store: StateStore, args) -> None:
    task_id = task["task_id"]
    args = task_evaluation_args(task, args)
    t0 = time.time()
    previous = store.get(task_id) or {}
    if not _finish_pending_cleanup(task_id, store, args, previous):
        logger.error(f"{task_id} ({_model_label(task)}): cache cleanup incomplete; task moved to queue tail")
        return

    out_dir = _create_output_dir(args.output_root, task_id)
    selected_benchmark = args.benchmark
    worker_id = getattr(args, "worker_id", "") or socket.gethostname()
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
        resume_evaluation=False,
        cleanup_download_dir=None,
        cleanup_evaluation_dir=None,
        note=None,
    )

    # init states 随机化种子 = 队列条目自带的 seed(公开可验证,见 select_init_seed)。
    init_seed = None
    if get_profile(selected_benchmark).randomization_manifest_path is not None:
        init_seed = args.axis_randomization_seed
    if selected_benchmark in ("libero", "libero_pro", "libero_pro_custom_1") and not args.no_init_randomization:
        init_seed = select_init_seed(task)

    summary, error = None, ""
    ref = None
    revision = None
    model_dir = None
    stage = "resolving"
    try:
        # Validate the queue contract before downloading a potentially large
        # checkpoint. run_evaluation validates again at its direct-call boundary.
        base_model = select_base_model(task, selected_benchmark)
        max_total_bytes = MODEL_MAX_BYTES[parse_backbone(base_model).model_family]
        _report_progress(client, task_id, "downloading", {}, worker_id)
        ref, revision, model_dir = resolve_model(task, args.download_dir, args.allow_local_model)
        if ref:
            stage = "downloading"
            logger.info(f"Downloading model {ref} at {revision or 'main'} to {model_dir}")
            download_with_retry(ref, revision, model_dir, args, max_total_bytes=max_total_bytes)
        check_local_model_size(model_dir, max_total_bytes)
        stage = "evaluating"
        _report_progress(client, task_id, "prechecking", {}, worker_id)
        logger.info(f"Evaluating model {ref} at {revision}")
        summary, error = run_evaluation(
            task,
            model_dir,
            out_dir,
            args,
            init_seed=init_seed,
            progress_callback=lambda stage, detail: _report_progress(client, task_id, stage, detail, worker_id),
        )
    except TaskInvalidatedError as e:
        _invalidate_task(store, task_id, e)
        return
    except EvalInterrupted:
        _schedule_clean_retry(
            store,
            task_id,
            args,
            "attempt interrupted",
            download_cache=model_dir if stage == "downloading" and ref else None,
            evaluation_cache=out_dir,
        )
        logger.warning(
            f"{task_id} ({_model_label(task)}): attempt interrupted; artifacts preserved at {out_dir}, task requeued"
        )
        return
    except ModelSizeExceeded as e:
        error = str(e)
        logger.error(f"{task_id} ({_model_label(task)}): {error}; reporting task rejected")
    except DownloadError as e:
        _schedule_clean_retry(
            store,
            task_id,
            args,
            f"download failed ({'permanent' if e.permanent else 'retryable'}): {e}",
            download_cache=model_dir if ref else None,
            evaluation_cache=out_dir,
        )
        logger.error(
            f"{task_id} ({_model_label(task)}): download failed; download cache cleared, "
            "task requeued, no score submitted"
        )
        return
    except EvalInfrastructureError as e:
        _schedule_clean_retry(
            store,
            task_id,
            args,
            f"evaluation incomplete: {e}",
            evaluation_cache=out_dir,
        )
        logger.error(
            f"{task_id} ({_model_label(task)}): evaluation incomplete: {e}; "
            f"artifacts preserved at {out_dir}, task requeued for a full run, no score submitted"
        )
        return
    except Exception as e:
        # ValueError 来自队列边界校验（非法 repo/commit），属于明确的输入拒绝，
        # 可以作为终态结论返回；执行/基础设施异常保留诊断产物后重试。
        if stage == "resolving" and isinstance(e, ValueError):
            error = f"invalid benchmark task: {e}"
            logger.error(f"{task_id} ({_model_label(task)}): {error}; reporting task rejected")
        else:
            _schedule_clean_retry(
                store,
                task_id,
                args,
                f"unexpected {stage} failure: {type(e).__name__}: {e}",
                download_cache=model_dir if stage == "downloading" and ref else None,
                evaluation_cache=out_dir,
            )
            logger.exception(
                f"{task_id} ({_model_label(task)}): unexpected {stage} failure; "
                f"artifacts preserved at {out_dir}, affected caches cleaned and task requeued, no score submitted"
            )
            return

    try:
        payload = build_score_payload(
            {**task, "benchmark": selected_benchmark, "protocol_revision": _protocol_revision(selected_benchmark)},
            summary,
            time.time() - t0,
            error,
            init_seed=init_seed,
            benchmark=selected_benchmark,
        )
        incomplete_reason = successful_payload_incomplete_reason(payload)
        if incomplete_reason:
            _schedule_clean_retry(
                store,
                task_id,
                args,
                f"score payload failed completeness gate: {incomplete_reason}",
                evaluation_cache=out_dir,
            )
            logger.error(
                f"{task_id} ({_model_label(task)}): score payload is incomplete ({incomplete_reason}); "
                f"artifacts preserved at {out_dir}, task requeued, no score submitted"
            )
            return
        (out_dir / "score_payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception as e:
        _schedule_clean_retry(
            store,
            task_id,
            args,
            f"score construction failed: {type(e).__name__}: {e}",
            evaluation_cache=out_dir,
        )
        logger.exception(
            f"{task_id} ({_model_label(task)}): score construction failed; "
            f"artifacts preserved at {out_dir}, task requeued, no score submitted"
        )
        return
    store.update(task_id, status="done_pending_submit", payload=payload)

    if not try_submit(client, store, task_id, payload):
        entry = store.get(task_id) or {}
        if entry.get("status") == "pending":
            logger.warning(f"{task_id} ({_model_label(task)}): incomplete score blocked; task will be re-evaluated")
        else:
            logger.warning(f"{task_id} ({_model_label(task)}): completed score retained; ledger backoff controls retry")


# ----------------------------------------------------------------------------
# 主循环:轮询 + 去重 + 补交
# ----------------------------------------------------------------------------
def classify_queued_task(entry: dict | None, task: dict, benchmark: str | None = None) -> str:
    """决定后端队列任务的处置(纯函数,供单测)。

    返回:
      "evaluate"  没见过的任务,或同 task_id 换了 repo/commit/base_model(miner 重新
                  提交)—— 入队评测。换 commit 时若旧结果尚未提交,直接作废:
                  后端只关心当前提交。
      "skip"      本地正在排队/评测，或同一提交已获后端确认。评分 POST
                  会创建 challenge attempt，并非幂等；已确认结果绝不自动重发。
    """
    if entry is None:
        return "evaluate"
    if entry.get("status") in ("pending", "running"):
        return "skip"
    if entry.get("status") == "stale":
        return "evaluate"
    prev = entry.get("task") or {}
    if benchmark is not None and entry.get("benchmark") != benchmark:
        return "evaluate"
    expected_revision = _protocol_revision(benchmark)
    if expected_revision is not None and entry.get("protocol_revision") != expected_revision:
        return "evaluate"
    if (task.get("hf_repo_id"), task.get("hf_commit"), task.get("base_model")) != (
        prev.get("hf_repo_id"),
        prev.get("hf_commit"),
        prev.get("base_model"),
    ):
        return "evaluate"
    if entry.get("status") == "submitted" and entry.get("payload"):
        if successful_payload_incomplete_reason(entry["payload"]):
            return "evaluate"
        return "skip"
    return "skip"  # done_pending_submit 走补交路径;abandoned 不再纠缠


def _handle_signal(_signum, _frame):
    # Event.set() also acquires a lock. Repeated SIGTERM (including uv's
    # forwarded signal) can interrupt that same lock and deadlock shutdown.
    global _shutdown_requested
    _shutdown_requested = True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend-url", required=True)
    p.add_argument(
        "--worker-id",
        default=os.environ.get("BENCHMARK_WORKER_ID", socket.gethostname()),
        help="进度上报中的 worker 标识(默认 BENCHMARK_WORKER_ID 或 hostname)",
    )
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
    p.add_argument(
        "--worker-key",
        default=os.environ.get("WORKER_KEY", ""),
        help="Worker key for the read-only AXIS rotation endpoint (WORKER_KEY)",
    )
    p.add_argument(
        "--axis-benchmark-dir",
        type=pathlib.Path,
        help="Generated AXIS releases; defaults to a separate .cache directory for each backend URL",
    )
    p.add_argument(
        "--axis-selector-root",
        type=pathlib.Path,
        help="Pinned selector checkout; enables automatic rotation together with --axis-runtime-pool",
    )
    p.add_argument("--axis-runtime-pool", type=pathlib.Path, help="Frozen runtime pool with adjacent task snapshots")

    p.add_argument("--once", action="store_true", help="只轮询一次,处理完即退出(调试用)")
    p.add_argument(
        "--state-file",
        default=str(VALIDATOR_ROOT / "benchmark_worker_state.json"),
        help="本地任务状态账本(去重与断点恢复的依据)",
    )
    p.add_argument("--output-root", default=str(VALIDATOR_ROOT / "eval_runs"), help="评测产物根目录")
    p.add_argument("--download-dir", default=str(VALIDATOR_ROOT / "hf_models"), help="HF 模型下载目录")
    p.add_argument(
        "--download-strategies",
        required=True,
        help='模型下载策略顺序:hfd-mirror,hfd,hub-mirror,hub。建议在中国大陆使用"hfd-mirror,hub-mirror";在国外使用"hfd,hub"',
    )

    ## Params below will be passed thourgh to run_eval.py
    p.add_argument("--num-trials", type=int, required=True)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument(
        "--workers-per-gpu",
        type=int,
        default=None,
        help="每张 GPU 的评测客户端数（默认：LIBERO 3，RoboTwin/Axis 1）",
    )
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
        help="policy server 实现(见 run_eval.py)。LingBot 在 4090 上实测 batched 配 "
        "--max-batch 4 --workers-per-gpu 8 吞吐最高,相对 upstream workers=6 约 +143%%;"
        "推理路径较新,生产默认保持 upstream",
    )
    p.add_argument(
        "--lingbot-server-impl",
        choices=("upstream", "batched", "static"),
        default=None,
        help="仅覆盖 LingBot 队列任务的 server 实现；static 使用固定 batch cohort 和严格确定性设置",
    )
    p.add_argument(
        "--max-batch",
        type=int,
        default=4,
        help="动态 batch 的上限，必须是 2 的幂（默认 4）",
    )
    benchmark_group = p.add_mutually_exclusive_group()
    benchmark_group.add_argument(
        "--axis-only",
        action="store_true",
        help="Accept only AXIS queue tasks, preserving each task's benchmark version",
    )
    benchmark_group.add_argument(
        "--benchmark",
        help="Override the queue benchmark; when omitted, use each task's benchmark",
    )
    benchmark_group.add_argument(
        "--axis_v1.0",
        dest="benchmark",
        action="store_const",
        const=AXIS_V1_NAME,
        help="Shortcut for --benchmark=axis_v1.0",
    )
    p.add_argument("--task-ids", default="", help="调试用:只评测这些 task id(逗号分隔)")
    p.add_argument(
        "--no-init-randomization",
        action="store_true",
        help="关闭 init states 混合随机化,退回全官方固定初始状态(仅调试/复现旧行为;"
        "默认用队列条目自带的 seed(公开可验证),一半 trial 用官方 init、一半用重采样 init,"
        "抗评测集过拟合)",
    )
    p.add_argument(
        "--eval-config",
        default=None,
        help="显式 openpi 训练配置名(默认根据 checkpoint 架构自动选择)",
    )
    p.add_argument(
        "--lingbot-norm-stats",
        default=None,
        help="覆盖 validator 内置的 LingBot-VLA 2.0 LIBERO normalization JSON(调试用)",
    )
    p.add_argument(
        "--evaluator-source-git-commit",
        default=os.environ.get("EVALUATOR_SOURCE_GIT_COMMIT"),
        help=(
            "Full immutable evaluator revision. Default: require a clean Git checkout and use HEAD; "
            "source bundles without .git must set this explicitly."
        ),
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
    if args.max_batch < 1 or args.max_batch & (args.max_batch - 1):
        p.error("--max-batch must be a positive power of two")
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
    args.axis_benchmark_dir = (args.axis_benchmark_dir or default_rotation_directory(args.backend_url)).resolve()
    if bool(args.axis_selector_root) != bool(args.axis_runtime_pool):
        p.error("--axis-selector-root and --axis-runtime-pool must be supplied together")
    if args.axis_selector_root:
        if not args.worker_key:
            p.error("automatic AXIS rotation requires --worker-key or WORKER_KEY")
        if args.benchmark is not None:
            p.error("automatic AXIS rotation uses queue benchmarks; remove the explicit benchmark override")
    configure_axis_profiles(args.axis_benchmark_dir)
    if args.benchmark is not None:
        try:
            get_profile(args.benchmark)
            _apply_benchmark_options(args)
        except ValueError as exc:
            p.error(str(exc))
    return args


def _apply_benchmark_options(args) -> None:
    if getattr(args, "workers_per_gpu", None) is None:
        args.workers_per_gpu = 1 if (args.benchmark == "robotwin" or is_axis_benchmark(args.benchmark)) else 3
    if args.benchmark == "libero_plus":
        if args.num_trials != 1:
            raise ValueError("libero_plus official protocol requires --num-trials 1")
        if args.task_ids:
            raise ValueError(
                "benchmark_worker cannot score a LIBERO-Plus --task-ids subset; use run_eval.py for development"
            )
        # Perturbation-specific init states are part of each registered variant.
        args.no_init_randomization = True
    if args.benchmark == "robotwin":
        if args.num_trials != 100:
            raise ValueError("robotwin official protocol requires --num-trials 100")
        if args.task_ids:
            raise ValueError(
                "benchmark_worker cannot score a RoboTwin --task-ids subset; use run_eval.py for development"
            )
        args.no_init_randomization = True
    if is_axis_benchmark(args.benchmark):
        trials = get_profile(args.benchmark).expected_trials_per_task or 1
        if args.num_trials != trials:
            raise ValueError(f"{args.benchmark} protocol requires --num-trials {trials}")
        if args.task_ids:
            raise ValueError(
                f"benchmark_worker cannot score an {args.benchmark} --task-ids subset; use run_eval.py for development"
            )
        if args.server_impl != "upstream":
            raise ValueError(f"{args.benchmark} currently requires --server-impl upstream")
        if args.eval_config not in (None, "pi05_axis_joint"):
            raise ValueError(f"{args.benchmark} requires --eval-config pi05_axis_joint")
        args.eval_config = "pi05_axis_joint"
        args.no_init_randomization = True


def preflight_runtime(benchmark: str) -> None:
    from libero_eval.paths import (
        AXIS_VENV_PY,
        CLIENT_VENV_PY,
        LINGBOT_VLA_V2_VENV_PY,
        ROBOTWIN_VENV_PY,
        SERVER_VENV_PY,
    )

    if benchmark == "robotwin":
        runtime_envs = [
            (LINGBOT_VLA_V2_VENV_PY, "LingBot-VLA 2.0 server venv"),
            (ROBOTWIN_VENV_PY, "RoboTwin simulator venv"),
        ]
    elif is_axis_benchmark(benchmark):
        runtime_envs = [
            (SERVER_VENV_PY, "OpenPI server venv"),
            (AXIS_VENV_PY, "AXIS simulator venv"),
        ]
    else:
        # The policy runtime is selected per queue task. run_eval performs its
        # family-specific preflight; only the shared LIBERO client is required
        # at worker startup.
        runtime_envs = [(CLIENT_VENV_PY, "LIBERO client venv")]
    for py, what in runtime_envs:
        if not py.exists():
            raise ValueError(f"Missing {py}; install the {what} first (see README).")


def main():
    args = parse_args()
    rotation = None
    try:
        args.evaluator_source_git_commit = resolve_evaluator_source_commit(args.evaluator_source_git_commit)
        if getattr(args, "axis_selector_root", None):
            rotation = AxisRotation(
                directory=args.axis_benchmark_dir,
                backend_url=args.backend_url,
                selector_root=args.axis_selector_root,
                runtime_pool=args.axis_runtime_pool,
            )
        if args.benchmark is not None:
            preflight_runtime(args.benchmark)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        sys.exit(f"[benchmark_worker] {exc}")
    log_path = _setup_logger(args.backend_url)
    _install_stderr_logging()
    logger.info(f"worker log: {log_path}")
    logger.info("benchmark source: %s", args.benchmark or "queue (per task)")
    if args.axis_only:
        logger.info("queue filter: AXIS only; non-AXIS tasks and pending scores are left untouched")

    client = BackendClient(
        args.backend_url,
        public_api_key=args.public_api_key,
        admin_api_key=args.admin_api_key,
        queue_path=args.queue_path,
        worker_api_key=getattr(args, "worker_key", None),
    )
    store = StateStore(pathlib.Path(args.state_file))
    local_q: queue.Queue = queue.Queue()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # 启动恢复:上次死在排队/评测中的任务重新入队;评完未确认的走补交路径。
    for task_id, entry in store.all_tasks().items():
        if entry.get("status") in ("pending", "running") and entry.get("task"):
            if not task_matches_filter(entry["task"], args):
                continue
            try:
                recovered_args = task_evaluation_args(entry["task"], args)
                preflight_runtime(recovered_args.benchmark)
            except ValueError as exc:
                store.update(task_id, status="stale", note=str(exc))
                logger.warning("recover: skip %s: %s", task_id, exc)
                continue
            logger.info(f"recover: requeue {task_id} ({_model_label(entry['task'])}, was {entry['status']})")
            store.update(task_id, status="pending")
            local_q.put(entry["task"])

    worker = threading.Thread(target=worker_loop, args=(local_q, client, store, args), daemon=True)
    worker.start()
    logger.info(f"benchmark worker up: backend={args.backend_url} poll={args.poll_interval:.0f}s once={args.once}")

    first_poll = True
    while not _stopping():
        # 清除的是触发“本轮立即轮询”的旧通知；本轮期间的新通知会保留到
        # 末尾 wait，避免恰好发生在 HTTP 请求期间的刷新信号被丢失。
        _poll_wakeup_event.clear()
        if _stopping():
            break
        # Prepare and publish complete bundles before a new baseline can be claimed.
        # Polling/generation failures do not mark queue submissions as failed.
        if rotation is not None:
            try:
                pending_rotation = client.fetch_rotation()
                if pending_rotation is not None:
                    ready = rotation.prepare(pending_rotation)
                    if ready is not None:
                        logger.info("AXIS rotation prepared: %s", ready)
            except (BackendError, OSError, ValueError, KeyError, TypeError, ImportError) as exc:
                logger.warning("AXIS rotation not ready; retry on next poll: %s", exc)
        refresh_axis_profiles()
        # 先补交欠账(worker 已放弃原地重试的、或重启前遗留的)。
        # _inflight 保证不会和 worker 线程对同一任务双重提交。
        with _inflight_lock:
            inflight = set(_inflight)
        for task_id, entry in store.all_tasks().items():
            if entry.get("status") == "done_pending_submit" and task_id not in inflight and submit_retry_due(entry):
                if not task_matches_filter(entry.get("task") or {}, args):
                    continue
                if not pending_score_matches_profile(entry, args.benchmark):
                    store.update(
                        task_id,
                        status="stale",
                        note="benchmark profile or protocol changed; old score will not be submitted",
                    )
                    continue
                incomplete_reason = successful_payload_incomplete_reason(entry["payload"])
                if incomplete_reason:
                    try_submit(client, store, task_id, entry["payload"])
                    if entry.get("task"):
                        local_q.put(entry["task"])
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
            try:
                selected_args = task_evaluation_args(t, args)
                preflight_runtime(selected_args.benchmark)
            except ValueError as exc:
                logger.warning("queue task %s (%s) skipped: %s", tid, _model_label(t), exc)
                continue
            selected_benchmark = selected_args.benchmark
            with _inflight_lock:
                if tid in _inflight:
                    continue  # worker 线程正处理中,下一轮再看
            entry = store.get(tid)
            verdict = classify_queued_task(entry, t, selected_benchmark)
            if verdict == "evaluate":
                if entry is not None:
                    prev = entry.get("task") or {}
                    logger.info(
                        f"{tid}: backend re-queued with new submission "
                        f"{prev.get('hf_repo_id')}@{str(prev.get('hf_commit'))[:12]} -> "
                        f"{t.get('hf_repo_id')}@{str(t.get('hf_commit'))[:12]}; re-evaluating"
                    )
                store.update(tid, status="pending", task=t, benchmark=selected_benchmark, payload=None)
                logger.info(
                    "queue task %s: benchmark=%s source=%s queue_benchmark=%r",
                    tid,
                    selected_benchmark,
                    "cli" if args.benchmark is not None else "queue",
                    t.get("benchmark"),
                )
                local_q.put(t)
                new += 1
        if new:
            logger.info(f"picked up {new} new task(s) from backend queue")
        elif first_poll and tasks is not None:
            # 首轮空转也出一条日志,否则启动后长时间静默像卡死。
            if tasks:
                logger.info(
                    f"queue has {len(tasks)} task(s), no new eligible tasks; "
                    f"see routing warnings and local ledger ({args.state_file})"
                )
            else:
                logger.info(f"queue empty; polling every {args.poll_interval:.0f}s")
        if tasks is not None:
            first_poll = False

        if args.once:
            local_q.put(None)
            break
        _wait_for_event(_poll_wakeup_event, args.poll_interval)

    worker.join()
    logger.info("benchmark worker stopped")


if __name__ == "__main__":
    main()
