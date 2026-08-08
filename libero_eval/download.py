"""多策略 HF 模型下载:hfd.sh(aria2c 并行)x 镜像/官方端点,链式回退。

策略 = (是否用 hfd.sh 预取, 是否走镜像端点) 的组合。官方路径由
snapshot_download 收尾；镜像路径按固定 commit 的文件清单逐文件校验 size，
并校验 LFS/Xet sha256 或普通 Git blob sha1，全程不依赖 huggingface.co。

顺序配置:环境变量 MODEL_DOWNLOAD_STRATEGIES(逗号分隔),或调用方显式传入
strategies(worker/run_eval 的 --download-strategies 参数)。

本模块自包含(仅 stdlib + 惰性导入 huggingface_hub),worker 以
`from libero_eval.download import ...`、run_eval 以 `from download import ...`
两种方式导入均可。
"""

from __future__ import annotations

import argparse
import errno
import fnmatch
import hashlib
import os
import pathlib
import re
import signal
import subprocess
import time

_VALIDATOR_ROOT = pathlib.Path(__file__).resolve().parent.parent

# 评测必须 pin 到具体的 HF commit(git SHA-1,完整 40 位小写十六进制)。
# 分支名/tag 会随内容漂移:同一引用先后解析到的文件可能不同,结果无法对应
# 到唯一提交,因此不接受。worker 与 run_eval 共用此定义。
COMMIT_HASH_RE = re.compile(r"^[0-9a-f]{40}$")

HFD_SCRIPT = _VALIDATOR_ROOT / "hfd.sh"
OFFICIAL_ENDPOINT = "https://huggingface.co"
DEFAULT_MIRROR_ENDPOINT = "https://hf-mirror.com"
DEFAULT_STRATEGIES = "hfd-mirror,hfd,hub"

# 策略名 -> (use_hfd, use_mirror)。hub 策略即"没有 hfd 预取步骤的同一条路径"。
_STRATEGY_TABLE: dict[str, tuple[bool, bool]] = {
    "hfd-mirror": (True, True),
    "hfd": (True, False),
    "hub-mirror": (False, True),
    "hub": (False, False),
}

# aria2c 单文件连接数(-x,hfd 上限 10;模型 repo 通常是单个大 safetensors,
# 连接数是主要加速项)与并发文件数(-j)。
_HFD_THREADS = 8
_HFD_JOBS = 5

# hfd.sh 断点续传信号:三处 "Re-run to resume"(下载不完整 exit 1、列文件失败
# exit 1、SIGINT trap exit 130)。hfd.sh 重跑会 diff 本地文件与 manifest 只下
# 缺失部分(aria2c -c),部分文件从不清理,因此命中且本次有净进度时原地重跑
# 同一策略,而不是丢弃镜像已下载的部分回退下一策略。
_RESUME_HINT = "Re-run to resume"
_DEFAULT_RESUME_RETRIES = 3

# 镜像会剥 Authorization 头(见 README),token 对镜像既无用又等于把密钥
# 送给第三方;hfd.sh 会从环境继承这些变量,走镜像时必须剥干净。
_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_USERNAME")


class DownloadError(Exception):
    """所有下载策略均失败;message 逐策略汇总错误。

    ``permanent`` 只表示本轮无需在同一份缓存上继续尝试（例如远端文件缺失
    或官方校验确认内容不一致）。worker 不会把下载异常当成评测结论；无论
    permanent 与否都会清掉该提交的下载目录并重新排队。磁盘满、网络中断等
    validator 侧故障更不能归咎于 miner。
    """

    def __init__(self, message: str, *, permanent: bool = False):
        super().__init__(message)
        self.permanent = permanent


class SnapshotIntegrityError(OSError):
    """远端清单与本地快照不一致；快照不能用于评测。"""


def parse_strategies(spec: str) -> list[str]:
    """解析逗号分隔的策略串;未知名/空/重复一律 fail fast。"""
    names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in _STRATEGY_TABLE]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    problems = []
    if not names:
        problems.append("empty")
    if unknown:
        problems.append(f"unknown {unknown}")
    if duplicated:
        problems.append(f"duplicated {duplicated}")
    if problems:
        raise ValueError(
            f"invalid download strategies {spec!r}: {'; '.join(problems)} (valid names: {', '.join(_STRATEGY_TABLE)})"
        )
    return names


def download_model(
    repo_id: str,
    local_dir: pathlib.Path,
    *,
    revision: str | None = None,
    strategies: list[str] | None = None,
    repo_type: str = "model",
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    log=print,
) -> pathlib.Path:
    """按策略链下载 repo 到 local_dir,返回 local_dir;全部失败抛 DownloadError。

    strategies 为 None 时读环境变量 MODEL_DOWNLOAD_STRATEGIES,再默认
    DEFAULT_STRATEGIES。repo_type="dataset" 供数据集 repo 复用。
    allow_patterns/ignore_patterns 非空时筛选下载路径；hfd.sh 不支持可靠的
    按目录校验，因此这类请求直接使用对应端点的 snapshot_download。
    """
    names = (
        strategies
        if strategies is not None
        else parse_strategies(os.environ.get("MODEL_DOWNLOAD_STRATEGIES", DEFAULT_STRATEGIES))
    )
    local_dir = pathlib.Path(local_dir).resolve()
    # 镜像-only 策略不能在开跑前仍然硬依赖官方端点，否则“使用镜像”只是
    # 下载数据走镜像，预检照样会在被墙的 huggingface.co 上白等超时。
    first_use_mirror = _STRATEGY_TABLE[names[0]][1]
    preflight_endpoint = _mirror_endpoint() if first_use_mirror else OFFICIAL_ENDPOINT
    _preflight_repo_check(
        repo_id,
        revision,
        repo_type,
        log,
        endpoint=preflight_endpoint,
        official=not first_use_mirror,
    )
    attempts: list[tuple[str, str]] = []
    task_permanent = False
    integrity_failure = False
    for name in names:
        use_hfd, use_mirror = _STRATEGY_TABLE[name]
        endpoint = _mirror_endpoint() if use_mirror else OFFICIAL_ENDPOINT
        t0 = time.monotonic()
        log(f"[download] {repo_id}@{revision or 'main'}: trying strategy '{name}' via {endpoint}")
        # 官方策略错误可据类型判定 miner 提交永久无效；镜像错误仍视为
        # validator/镜像侧故障，避免把第三方镜像异常错误归因给 miner。
        failure_official = not use_mirror
        try:
            if use_hfd and not allow_patterns and not ignore_patterns:
                _run_hfd(repo_id, revision, local_dir, endpoint, repo_type, log)
                if use_mirror:
                    # 镜像的 repo_info(files_metadata=True) 已包含固定 commit 下
                    # 每个文件的 size、Git blob id 与 LFS sha256。直接以它为清单
                    # 做本地逐文件验全，不再为了“收尾”访问 huggingface.co。
                    _verify_local_snapshot(repo_id, revision, local_dir, endpoint, True, repo_type, log=log)
                else:
                    failure_official = True
                    _snapshot(repo_id, revision, local_dir, OFFICIAL_ENDPOINT, False, repo_type)
            else:
                if allow_patterns or ignore_patterns:
                    _snapshot(
                        repo_id,
                        revision,
                        local_dir,
                        endpoint,
                        use_mirror,
                        repo_type,
                        allow_patterns=allow_patterns,
                        ignore_patterns=ignore_patterns,
                    )
                else:
                    _snapshot(repo_id, revision, local_dir, endpoint, use_mirror, repo_type)
                if use_mirror:
                    # snapshot_download 对镜像/Xet 响应头兼容性不能作为完整性
                    # 边界；成功返回后仍以固定 commit 的清单重新验全。
                    _verify_local_snapshot(
                        repo_id,
                        revision,
                        local_dir,
                        endpoint,
                        True,
                        repo_type,
                        allow_patterns=allow_patterns,
                        ignore_patterns=ignore_patterns,
                        log=log,
                    )
            log(f"[download] strategy '{name}' OK in {time.monotonic() - t0:.0f}s -> {local_dir}")
            return local_dir
        except Exception as e:  # noqa: BLE001 - 每个策略的失败都要兜住并汇总
            msg = f"{type(e).__name__}: {e}"
            attempts.append((name, msg))
            log(f"[download] strategy '{name}' FAILED after {time.monotonic() - t0:.0f}s: {msg}")
            integrity_failure = integrity_failure or _is_integrity_mismatch(e)
            task_permanent = task_permanent or _is_task_permanent(e, official=failure_official)
            if _is_permanent(e, official=failure_official):
                log("[download] error is permanent (no strategy can succeed); aborting chain")
                break
    detail = "\n".join(f"  - {n}: {m}" for n, m in attempts)
    if integrity_failure:
        headline = (
            f"model artifact integrity mismatch for {repo_id}@{revision or 'main'}; "
            "checkpoint is not evaluable (mismatch details below)"
        )
    else:
        headline = f"all download strategies failed for {repo_id}@{revision or 'main'}"
    raise DownloadError(
        f"{headline}:\n{detail}",
        permanent=task_permanent,
    )


def _preflight_repo_check(
    repo_id: str,
    revision: str | None,
    repo_type: str,
    log,
    *,
    endpoint: str = OFFICIAL_ENDPOINT,
    official: bool = True,
) -> None:
    """策略链开跑前确认 repo@revision 存在。

    官方端点确认不存在/被 gate 属永久失败,直接抛 DownloadError；镜像端点
    的 401/404 可能是剥 token 或内容滞后，只记日志并让策略链继续。
    预检自身的临时故障(网络抖动等)只记日志放行,交给策略链自己碰运气。
    """
    from huggingface_hub import HfApi  # lazy: 本仓库唯一第三方依赖

    try:
        token = None if official else False
        HfApi(endpoint=endpoint, token=token).repo_info(
            repo_id=repo_id, repo_type=repo_type, revision=revision or "main"
        )
    except Exception as e:  # noqa: BLE001 - 永久/临时在此分流
        if _is_permanent(e, official=official):
            raise DownloadError(
                f"{repo_id}@{revision or 'main'} not found on {endpoint} ({type(e).__name__}: {e})",
                permanent=_is_task_permanent(e, official=official),
            ) from e
        log(f"[download] preflight repo check failed ({type(e).__name__}: {e}); trying strategies anyway")


def _matches_patterns(path: str, allow_patterns: list[str] | None, ignore_patterns: list[str] | None) -> bool:
    if allow_patterns and not any(fnmatch.fnmatchcase(path, pattern) for pattern in allow_patterns):
        return False
    if ignore_patterns and any(fnmatch.fnmatchcase(path, pattern) for pattern in ignore_patterns):
        return False
    return True


def _hash_file(path: pathlib.Path, algorithm: str, *, git_blob_size: int | None = None) -> str:
    digest = hashlib.new(algorithm)
    if git_blob_size is not None:
        digest.update(f"blob {git_blob_size}\0".encode())
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_local_snapshot(
    repo_id: str,
    revision: str | None,
    local_dir: pathlib.Path,
    endpoint: str,
    use_mirror: bool,
    repo_type: str,
    *,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    log=print,
) -> None:
    """按固定 commit 的远端清单逐文件验证本地快照，不下载任何内容。

    普通 Git 文件校验 Git blob SHA-1；LFS/Xet 文件校验 LFS SHA-256。
    size 先全量检查，任何缺失/大小错误都立即失败，避免对已知坏快照再读
    数十 GB 做哈希。镜像端只用于取得清单，内容正确性最终由哈希决定。
    """
    from huggingface_hub import HfApi  # lazy: 本仓库唯一第三方依赖

    token = False if use_mirror else None
    info = HfApi(endpoint=endpoint, token=token).repo_info(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision or "main",
        files_metadata=True,
    )
    if revision and info.sha != revision:
        raise SnapshotIntegrityError(f"repo metadata resolved to {info.sha}, expected pinned commit {revision}")

    entries = [item for item in info.siblings if _matches_patterns(item.rfilename, allow_patterns, ignore_patterns)]
    if not entries:
        raise SnapshotIntegrityError("remote manifest is empty after applying download patterns")

    local_dir = pathlib.Path(local_dir).resolve()
    size_errors: list[str] = []
    checked: list[tuple[object, pathlib.Path, int]] = []
    expected_paths: set[str] = set()
    expected_total = 0
    for item in entries:
        rel = pathlib.PurePosixPath(item.rfilename)
        if rel.is_absolute() or ".." in rel.parts:
            raise SnapshotIntegrityError(f"unsafe path in remote manifest: {item.rfilename!r}")
        if item.size is None:
            raise SnapshotIntegrityError(f"remote manifest has no size for {item.rfilename}")
        expected_size = int(item.size)
        path = local_dir.joinpath(*rel.parts)
        expected_paths.add(rel.as_posix())
        expected_total += expected_size
        try:
            actual_size = path.stat().st_size
        except FileNotFoundError:
            size_errors.append(f"missing {item.rfilename} (expected {expected_size} bytes)")
            continue
        if actual_size != expected_size:
            size_errors.append(f"size {item.rfilename}: expected {expected_size}, got {actual_size}")
            continue
        checked.append((item, path, expected_size))

    # .hfd/ 是 hfd.sh 的 manifest/log，.cache/ 是 huggingface_hub 的本地
    # 元数据；除此以外的额外文件会改变模型目录语义，必须拒绝。
    for dirpath, dirnames, filenames in os.walk(local_dir):
        rel_dir = pathlib.Path(dirpath).relative_to(local_dir)
        if rel_dir == pathlib.Path("."):
            dirnames[:] = [name for name in dirnames if name not in {".hfd", ".cache"}]
        for filename in filenames:
            rel_path = (rel_dir / filename).as_posix()
            if rel_path not in expected_paths:
                size_errors.append(f"unexpected {rel_path}")

    if size_errors:
        detail = "\n".join(f"  - {problem}" for problem in size_errors[:20])
        if len(size_errors) > 20:
            detail += f"\n  - ... and {len(size_errors) - 20} more"
        raise SnapshotIntegrityError(f"snapshot size verification failed:\n{detail}")

    log(f"[download] verifying {len(checked)} files ({expected_total} bytes) against {endpoint} manifest")
    hash_errors: list[str] = []
    for item, path, expected_size in checked:
        lfs = getattr(item, "lfs", None)
        if lfs is not None and getattr(lfs, "sha256", None):
            expected_hash = lfs.sha256
            actual_hash = _hash_file(path, "sha256")
            kind = "LFS sha256"
        elif getattr(item, "blob_id", None):
            expected_hash = item.blob_id
            actual_hash = _hash_file(path, "sha1", git_blob_size=expected_size)
            kind = "Git blob sha1"
        else:
            raise SnapshotIntegrityError(f"remote manifest has no content hash for {item.rfilename}")
        if actual_hash != expected_hash:
            hash_errors.append(f"{kind} {item.rfilename}: expected {expected_hash}, got {actual_hash}")

    if hash_errors:
        detail = "\n".join(f"  - {problem}" for problem in hash_errors[:20])
        if len(hash_errors) > 20:
            detail += f"\n  - ... and {len(hash_errors) - 20} more"
        raise SnapshotIntegrityError(f"snapshot content verification failed:\n{detail}")
    log(f"[download] local snapshot verified: {len(checked)} files, {expected_total} bytes")


def _mirror_endpoint() -> str:
    # 调用时读取(不在 import 时固化),方便测试与运行中切换。
    return os.environ.get("HF_MIRROR_ENDPOINT", DEFAULT_MIRROR_ENDPOINT)


def _build_hfd_cmd(
    repo_id: str,
    revision: str | None,
    local_dir: pathlib.Path,
    endpoint: str,
    repo_type: str,
    send_token: bool,
) -> tuple[list[str], dict[str, str]]:
    """拼 hfd.sh 的 argv 与子进程 env(纯函数,供单测)。

    token 只走环境变量不拼 argv(hfd.sh 本就继承 HF_TOKEN/HF_USERNAME,
    argv 会把 token 暴露在 ps 里)。
    """
    cmd = [
        "bash",
        str(HFD_SCRIPT),
        repo_id,
        "--tool",
        "aria2c",
        "-x",
        str(_HFD_THREADS),
        "-j",
        str(_HFD_JOBS),
        "--local-dir",
        str(local_dir),
    ]
    if revision:
        cmd += ["--revision", revision]
    if repo_type == "dataset":
        cmd += ["--dataset"]
    env = dict(os.environ)
    env["HF_ENDPOINT"] = endpoint
    if not send_token:
        for k in _TOKEN_ENV_VARS:
            env.pop(k, None)
    return cmd, env


def _is_resumable(output: str) -> bool:
    """本次 hfd.sh 输出是否为可续传失败(纯函数,供单测)。

    输出带 ANSI 色码,但 "Re-run to resume" 字面子串完整,substring 匹配即可。
    """
    return _RESUME_HINT in output


def _dir_size(root: pathlib.Path) -> int:
    """local_dir 下已落盘字节数,排除 .hfd/ 元数据(console.log/download.log
    每次重跑必增长,计入会让进度门失效)。文件可能在遍历中消失(aria2 下载完
    删 .aria2 sidecar),stat 失败按 0 计;目录不存在则为 0。
    """
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".hfd"]
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total


def _run_hfd(
    repo_id: str,
    revision: str | None,
    local_dir: pathlib.Path,
    endpoint: str,
    repo_type: str,
    log,
) -> None:
    if not HFD_SCRIPT.is_file():
        raise FileNotFoundError(f"hfd.sh not found at {HFD_SCRIPT}")
    cmd, env = _build_hfd_cmd(
        repo_id,
        revision,
        local_dir,
        endpoint,
        repo_type,
        send_token=(endpoint == OFFICIAL_ENDPOINT),
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / ".hfd").mkdir(exist_ok=True)
    # hfd 进度是 \r 单行刷新,透传会刷屏;落文件,失败时 tail 并入异常。
    console_log = local_dir / ".hfd" / "console.log"
    timeout = float(os.environ.get("MODEL_DOWNLOAD_TIMEOUT", "7200"))
    # 可续传失败且本次有净进度时原地重跑,上限 = 首跑之外的重跑次数。
    # max(0,...) 防负值把 range 掏空、一次都不跑;垃圾值同 TIMEOUT 一样 fail fast。
    # TimeoutError 不重跑:一次已耗满整个预算。
    retries = max(0, int(os.environ.get("MODEL_DOWNLOAD_RESUME_RETRIES", str(_DEFAULT_RESUME_RETRIES))))
    log(f"[download] hfd.sh console -> {console_log}")
    for attempt in range(retries + 1):
        # console.log 是 append 模式,记 offset 才能只读本次输出。
        offset = console_log.stat().st_size if console_log.exists() else 0
        bytes_before = _dir_size(local_dir)
        with open(console_log, "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.strftime('%F %T')} endpoint={endpoint} rev={revision or 'main'} =====\n")
            f.flush()
            # start_new_session=True: pid 即 pgid,超时可杀掉 hfd 及其 aria2c 子进程。
            proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                ret = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise TimeoutError(
                    f"hfd.sh timed out after {timeout:.0f}s (partial files kept; next strategy resumes them)"
                ) from None
        if ret == 0:
            return
        # offset 是字节数,必须二进制 seek 再解码(文本模式 seek 任意偏移未定义)。
        with open(console_log, "rb") as f:
            f.seek(offset)
            out = f.read().decode("utf-8", errors="replace")
        bytes_after = _dir_size(local_dir)
        # 无进度的可续传失败(如列文件失败)立即回退,不白送重试:列表阶段的
        # 断点文件在被排除的 .hfd/ 下,进度门看不见;对死镜像重试最坏多烧一个
        # 整 timeout。
        if attempt < retries and _is_resumable(out) and bytes_after > bytes_before:
            log(
                f"[download] hfd incomplete but resumable ({bytes_before} -> {bytes_after} bytes); "
                f"re-running same strategy (resume attempt {attempt + 1}/{retries})"
            )
            continue
        tail = "\n".join(out.splitlines()[-5:])
        raise RuntimeError(f"hfd.sh exited {ret}; log tail:\n{tail}")


def _snapshot(
    repo_id: str,
    revision: str | None,
    local_dir: pathlib.Path,
    endpoint: str,
    use_mirror: bool,
    repo_type: str,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> None:
    from huggingface_hub import HfApi, snapshot_download  # lazy: 本仓库唯一第三方依赖

    # 镜像会剥 Authorization 头,token=False 阻止 huggingface_hub 把本地
    # token 发给第三方端点;官方端点走默认 token 解析。
    token = False if use_mirror else None
    # 先显式取 repo_info:
    # 1) 404/401/gated 变成类型化异常向上抛(策略链据此判永久),而不是被
    #    snapshot_download 的"API 失败但 local_dir 非空则沿用本地文件"回退
    #    吞掉——hfd 残留的 .hfd/ 会让无效目录被当成有效快照静默返回;
    # 2) 把 revision 钉到具体 commit,消除解析与下载之间分支漂移的竞态。
    info = HfApi(endpoint=endpoint, token=token).repo_info(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision or "main",
        files_metadata=True,
    )
    kwargs = dict(
        repo_id=repo_id,
        revision=info.sha,
        repo_type=repo_type,
        local_dir=str(local_dir),
        endpoint=endpoint,
        token=token,
    )
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns
    try:
        snapshot_download(**kwargs)
    except Exception as exc:  # noqa: BLE001 - 给第三方下载器的完整性错误补齐上下文
        if _is_integrity_mismatch(exc):
            raise _enrich_integrity_mismatch(repo_id, info.sha, info.siblings, exc) from exc
        raise


_INTEGRITY_ERROR_MARKERS = (
    "consistency check failed",  # huggingface_hub legacy HTTP downloader
    "file size mismatch",  # hf_xet: expected N bytes but downloaded M bytes
    "checksum mismatch",
    "sha256 mismatch",
)
_SIZE_MISMATCH_RE = re.compile(
    r"expected\s+(?P<expected>\d+)(?:\s+bytes?)?\s+but\s+"
    r"(?:downloaded|got|has\s+size)\s+(?P<actual>\d+)(?:\s+bytes?)?",
    re.IGNORECASE,
)
_LEGACY_SIZE_MISMATCH_RE = re.compile(
    r"should\s+be\s+of\s+size\s+(?P<expected>\d+)\s+but\s+has\s+size\s+(?P<actual>\d+)",
    re.IGNORECASE,
)


def _is_integrity_mismatch(exc: Exception) -> bool:
    """下载器是否已给出具体的文件 size/hash 不一致证据。"""
    if isinstance(exc, SnapshotIntegrityError):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _INTEGRITY_ERROR_MARKERS)


def _enrich_integrity_mismatch(repo_id: str, revision: str, siblings, exc: Exception) -> SnapshotIntegrityError:
    """把第三方下载器的 mismatch 统一为含 commit/path/expected/actual 的错误。"""
    message = str(exc)
    match = _SIZE_MISMATCH_RE.search(message) or _LEGACY_SIZE_MISMATCH_RE.search(message)
    if match is None:
        return SnapshotIntegrityError(
            f"artifact integrity mismatch for {repo_id}@{revision}; downloader error: {type(exc).__name__}: {message}"
        )

    expected = int(match.group("expected"))
    actual = int(match.group("actual"))
    candidates = sorted(
        item.rfilename for item in siblings if getattr(item, "size", None) is not None and int(item.size) == expected
    )
    if len(candidates) == 1:
        file_context = f"file={candidates[0]}"
    elif candidates:
        shown = ", ".join(candidates[:5])
        suffix = f", ... and {len(candidates) - 5} more" if len(candidates) > 5 else ""
        file_context = f"candidate_files=[{shown}{suffix}]"
    else:
        file_context = "file=unknown (no manifest entry has the expected size)"
    return SnapshotIntegrityError(
        f"artifact size mismatch for {repo_id}@{revision}: {file_context}, "
        f"expected={expected} bytes, actual={actual} bytes; downloader error: {type(exc).__name__}: {message}"
    )


def _is_task_permanent(exc: Exception, official: bool) -> bool:
    """判定仓库/commit 自身是否不可恢复，供 worker 决定是否报告 failed。"""
    # mismatch 是具体的数据完整性证据，不是 network unreachable。镜像策略
    # 仍会继续回退到官方策略交叉验证，但如果整条策略链最终失败，worker
    # 必须报告该 commit failed，而不能按基础设施故障无限重新排队。
    if _is_integrity_mismatch(exc):
        return True
    if not official:
        return False

    # 注意用 RemoteEntryNotFoundError 而非 EntryNotFoundError:后者的子类
    # LocalEntryNotFoundError 表示"离线且本地无缓存",是临时故障。
    from huggingface_hub.errors import (
        GatedRepoError,
        RemoteEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    return isinstance(
        exc,
        (RepositoryNotFoundError, RevisionNotFoundError, GatedRepoError, RemoteEntryNotFoundError),
    )


def _is_permanent(exc: Exception, official: bool) -> bool:
    """判定"换策略也没救"的错误:仓库永久错误或 validator 磁盘满。

    hfd.sh 子进程失败只有 exit code,不可细分,一律不算永久;镜像端点的
    404/401 可能只是镜像剥 token 或内容滞后,也不算。
    """
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return True
    # 镜像发现 mismatch 时保留官方交叉验证机会；官方也确认 mismatch 后
    # 立即短路。无论来源，_is_task_permanent 都会让最终结果明确失败。
    if _is_integrity_mismatch(exc):
        return official
    return _is_task_permanent(exc, official=official)


def _main() -> None:
    parser = argparse.ArgumentParser(description="多策略下载 HF repo(验证入口)")
    parser.add_argument("repo_id", help="HF repo,如 user/repo")
    parser.add_argument("--revision", default=None, help="commit/branch/tag,默认 main")
    parser.add_argument("--dest", default=None, help="落盘目录,默认 hf_models/<user>__<repo>")
    parser.add_argument(
        "--strategies",
        default=os.environ.get("MODEL_DOWNLOAD_STRATEGIES", DEFAULT_STRATEGIES),
        help=f"逗号分隔的策略顺序,可选: {', '.join(_STRATEGY_TABLE)}",
    )
    parser.add_argument("--dataset", action="store_true", help="下载 dataset 而非 model")
    args = parser.parse_args()

    strategies = parse_strategies(args.strategies)
    dest = pathlib.Path(args.dest or _VALIDATOR_ROOT / "hf_models" / args.repo_id.replace("/", "__"))
    download_model(
        args.repo_id,
        dest,
        revision=args.revision,
        strategies=strategies,
        repo_type="dataset" if args.dataset else "model",
    )


if __name__ == "__main__":
    _main()
