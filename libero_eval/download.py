"""多策略 HF 模型下载:hfd.sh(aria2c 并行)x 镜像/官方端点,链式回退。

策略 = (是否用 hfd.sh 预取, 是否走镜像端点) 的组合;每条成功路径最终都以
snapshot_download 收尾(对已落盘的 LFS 文件做 sha256 比对而不重下,并补齐
.cache/huggingface 元数据),正确性的单一事实来源是 huggingface_hub。

顺序配置:环境变量 MODEL_DOWNLOAD_STRATEGIES(逗号分隔),或调用方显式传入
strategies(worker/run_eval 的 --download-strategies 参数)。

本模块自包含(仅 stdlib + 惰性导入 huggingface_hub),worker 以
`from libero_eval.download import ...`、run_eval 以 `from download import ...`
两种方式导入均可。
"""

from __future__ import annotations

import argparse
import errno
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
    """所有下载策略均失败;message 逐策略汇总错误。"""


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
    log=print,
) -> pathlib.Path:
    """按策略链下载 repo 到 local_dir,返回 local_dir;全部失败抛 DownloadError。

    strategies 为 None 时读环境变量 MODEL_DOWNLOAD_STRATEGIES,再默认
    DEFAULT_STRATEGIES。repo_type="dataset" 供数据集 repo 复用。
    """
    names = (
        strategies
        if strategies is not None
        else parse_strategies(os.environ.get("MODEL_DOWNLOAD_STRATEGIES", DEFAULT_STRATEGIES))
    )
    local_dir = pathlib.Path(local_dir).resolve()
    _preflight_repo_check(repo_id, revision, repo_type, log)
    attempts: list[tuple[str, str]] = []
    for name in names:
        use_hfd, use_mirror = _STRATEGY_TABLE[name]
        endpoint = _mirror_endpoint() if use_mirror else OFFICIAL_ENDPOINT
        t0 = time.monotonic()
        log(f"[download] {repo_id}@{revision or 'main'}: trying strategy '{name}' via {endpoint}")
        try:
            if use_hfd:
                _run_hfd(repo_id, revision, local_dir, endpoint, repo_type, log)
                # 收尾/校验固定走官方端点:hf-mirror 对 resolve 的重定向响应缺
                # X-Repo-Commit/ETag 头,经镜像的元数据检查会退化成"沿用本地
                # 文件"而非哈希校验(实测)。官方端点只发轻量 HEAD;hfd 下好的
                # LFS 文件 sha256 一致则不重下,顺带补齐 .cache/huggingface
                # 元数据,后续重跑秒级通过。
                _snapshot(repo_id, revision, local_dir, OFFICIAL_ENDPOINT, False, repo_type)
            else:
                _snapshot(repo_id, revision, local_dir, endpoint, use_mirror, repo_type)
            log(f"[download] strategy '{name}' OK in {time.monotonic() - t0:.0f}s -> {local_dir}")
            return local_dir
        except Exception as e:  # noqa: BLE001 - 每个策略的失败都要兜住并汇总
            msg = f"{type(e).__name__}: {e}"
            attempts.append((name, msg))
            log(f"[download] strategy '{name}' FAILED after {time.monotonic() - t0:.0f}s: {msg}")
            if _is_permanent(e, official=not use_mirror):
                log("[download] error is permanent (no strategy can succeed); aborting chain")
                break
    detail = "\n".join(f"  - {n}: {m}" for n, m in attempts)
    raise DownloadError(f"all download strategies failed for {repo_id}@{revision or 'main'}:\n{detail}")


def _preflight_repo_check(repo_id: str, revision: str | None, repo_type: str, log) -> None:
    """策略链开跑前,先到官方端点确认 repo@revision 存在。

    repo 不存在/被 gate 属永久失败,直接抛 DownloadError,一个策略都不用试
    (存在性只信官方端点:镜像的 401/404 可能是剥 token 或内容滞后)。
    预检自身的临时故障(网络抖动等)只记日志放行,交给策略链自己碰运气。
    """
    from huggingface_hub import HfApi  # lazy: 本仓库唯一第三方依赖

    try:
        HfApi(endpoint=OFFICIAL_ENDPOINT).repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision or "main")
    except Exception as e:  # noqa: BLE001 - 永久/临时在此分流
        if _is_permanent(e, official=True):
            raise DownloadError(
                f"{repo_id}@{revision or 'main'} not found on {OFFICIAL_ENDPOINT} ({type(e).__name__}: {e})"
            ) from e
        log(f"[download] preflight repo check failed ({type(e).__name__}: {e}); trying strategies anyway")


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
        repo_id=repo_id, repo_type=repo_type, revision=revision or "main"
    )
    snapshot_download(
        repo_id=repo_id,
        revision=info.sha,
        repo_type=repo_type,
        local_dir=str(local_dir),
        endpoint=endpoint,
        token=token,
    )


def _is_permanent(exc: Exception, official: bool) -> bool:
    """判定"换策略也没救"的错误:只信官方端点的类型化异常与磁盘满。

    hfd.sh 子进程失败只有 exit code,不可细分,一律不算永久;镜像端点的
    404/401 可能只是镜像剥 token 或内容滞后,也不算。
    """
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
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
