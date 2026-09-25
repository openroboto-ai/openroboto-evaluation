"""
后端 Benchmark API 的最小 HTTP 客户端 | Minimal HTTP client for the backend benchmark API.

只覆盖 worker 需要的接口(prototype backend 协议):

    GET  /api/v1/benchmark/queue          -> 待评测任务队列(public key)
    GET  /api/v1/benchmark/rotation       -> 待准备的 AXIS 版本(worker key，可选)
    GET  /api/submission/{id}             -> 核对评分是否已经落库(public key)
    POST /api/v1/benchmark/task/{id}/score     -> 提交评测结果
    POST /api/benchmark-progress           -> 上报非终态评测进度

读接口和写接口使用不同的 key。不再使用旧 `/api/pending-tasks`。

仅用标准库(urllib),保持通信层零第三方依赖。
"""

import http.client
import json
import urllib.error
import urllib.request

USER_AGENT = "validator-benchmark-worker/0.1"
DEFAULT_QUEUE_PATH = "/api/v1/benchmark/queue"

# validator 内部沿用与运行产物一致的 ``evaluating``；prototype progress
# API 对应的 wire value 是 ``running``。其余合法值保持同名。
_PROGRESS_STAGE_MAP = {
    "downloading": "downloading",
    "prechecking": "prechecking",
    "evaluating": "running",
    "running": "running",
    "done": "done",
    "failed": "failed",
}

TASK_INVALIDATION_CODES = frozenset({"SUPERSEDED", "REJECTED"})


class BackendError(Exception):
    """与后端通信失败(HTTP 或网络层)。

    `status` 为 HTTP 状态码,网络层错误时为 None。
    `permanent` 标记重试无意义的错误(4xx;但 401/403/429 除外)。
    """

    def __init__(
        self,
        message: str,
        status: int | None = None,
        *,
        code: str | None = None,
        detail: str | None = None,
        request_id: str | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.detail = detail
        self.request_id = request_id

    @property
    def permanent(self) -> bool:
        # 401/403 是本端 key 没配对(后端并没有否定结果本身),改对 key 重试
        # 即可成功;429 是限流。两者都不能作为丢弃已评测结果的依据。
        return self.status is not None and 400 <= self.status < 500 and self.status not in (401, 403, 429)


class TaskInvalidatedError(BackendError):
    """POST 告知当前提交已作废；调用方必须停止评测并重新拉取队列。"""


class BackendClient:
    def __init__(
        self,
        base_url: str,
        public_api_key: str,
        admin_api_key: str | None = None,
        timeout_s: float = 30.0,
        submit_timeout_s: float = 300.0,
        queue_path: str = DEFAULT_QUEUE_PATH,
        worker_api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.public_api_key = public_api_key
        # Compatibility for callers written before keys were split. New
        # production callers pass both values explicitly.
        self.admin_api_key = public_api_key if admin_api_key is None else admin_api_key
        self.timeout_s = timeout_s
        self.submit_timeout_s = submit_timeout_s
        self.queue_path = "/" + queue_path.lstrip("/")
        self.worker_api_key = worker_api_key

    def _request(
        self,
        method: str,
        path: str,
        *,
        api_key: str,
        body: dict | None = None,
        timeout_s: float | None = None,
    ) -> dict:
        # 显式标识客户端；Cloudflare 会以 1010 拒绝 Python-urllib 默认 UA。
        headers = {"User-Agent": USER_AGENT}
        if api_key:
            headers["X-API-Key"] = api_key
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s if timeout_s is None else timeout_s) as resp:
                return json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            try:
                full_response_text = e.read().decode(errors="replace")
            except (OSError, http.client.HTTPException, UnicodeDecodeError):
                full_response_text = ""
            response_text = full_response_text[:500]
            error_body = {}
            try:
                decoded = json.loads(full_response_text)
                if isinstance(decoded, dict):
                    error_body = decoded
            except json.JSONDecodeError:
                pass
            code = error_body.get("code") if isinstance(error_body.get("code"), str) else None
            detail = error_body.get("detail") if isinstance(error_body.get("detail"), str) else None
            request_id = error_body.get("request_id") if isinstance(error_body.get("request_id"), str) else None
            error_type = (
                TaskInvalidatedError
                if method == "POST" and e.code == 409 and code in TASK_INVALIDATION_CODES
                else BackendError
            )
            raise error_type(
                f"{method} {path} -> HTTP {e.code}: {response_text}",
                status=e.code,
                code=code,
                detail=detail,
                request_id=request_id,
            ) from e
        except urllib.error.URLError as e:
            raise BackendError(f"{method} {path} -> {e.reason}") from e
        except http.client.HTTPException as e:
            # urllib lets response framing errors escape directly from
            # http.client. In particular, IncompleteRead means the peer
            # advertised a longer body than it delivered. This is a
            # transient transport failure, not a reason to terminate the
            # long-running worker.
            raise BackendError(f"{method} {path} -> incomplete or invalid HTTP response: {e}") from e
        except OSError as e:
            # Python 3.14 的 socket read timeout 会从 urllib 直接冒成
            # TimeoutError(OSError),而不是包装成 URLError。
            raise BackendError(f"{method} {path} -> {e}") from e
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            # A complete HTTP response can still be truncated/corrupted at
            # the payload layer. Keep all response decoding failures inside
            # the same retryable BackendError boundary as network failures.
            raise BackendError(f"{method} {path} -> invalid JSON response: {e}") from e

    def fetch_queue(self) -> list[dict]:
        """拉取后端待评测任务列表(可能包含本地已处理过的任务,由调用方去重)。

        响应体是 {"queue_size": N, "tasks": [...]} 信封(api_reference_zh.md 3.1),
        这里解开只返回任务列表。
        """
        data = self._request("GET", self.queue_path, api_key=self.public_api_key)
        tasks = data.get("tasks", []) if isinstance(data, dict) else data
        if not isinstance(tasks, list) or any(not isinstance(task, dict) for task in tasks):
            raise BackendError(f"GET {self.queue_path} -> invalid queue response: expected a list of task objects")
        return tasks

    def fetch_submission(self, task_id: str) -> dict:
        """读取任务详情,用于确认超时/5xx 的评分 POST 是否实际已经落库。"""
        return self._request("GET", f"/api/submission/{task_id}", api_key=self.public_api_key)

    def fetch_rotation(self) -> dict | None:
        """Read the pending rotation without claiming a queue submission."""
        if not self.worker_api_key:
            raise BackendError("AXIS rotation requires --worker-key or WORKER_KEY")
        response = self._request("GET", "/api/v1/benchmark/rotation", api_key=self.worker_api_key)
        if not isinstance(response, dict) or "data" not in response:
            raise BackendError("invalid rotation response: expected an object with data")
        rotation = response["data"]
        if rotation is not None and not isinstance(rotation, dict):
            raise BackendError("invalid rotation response: data must be an object or null")
        return rotation

    def submit_score(self, task_id: str, payload: dict) -> dict:
        # 后端同步落库耗时可能明显长于普通 GET,单独留出充足超时。
        return self._request(
            "POST",
            f"/api/v1/benchmark/task/{task_id}/score",
            api_key=self.admin_api_key,
            body=payload,
            timeout_s=self.submit_timeout_s,
        )

    def report_progress(self, task_id: str, stage: str, detail: dict, worker_id: str) -> dict:
        """Report the current non-terminal worker stage for a benchmark task."""
        try:
            wire_stage = _PROGRESS_STAGE_MAP[stage]
        except KeyError:
            raise ValueError(f"invalid benchmark progress stage: {stage!r}") from None
        return self._request(
            "POST",
            "/api/benchmark-progress",
            api_key=self.admin_api_key,
            body={
                "task_id": task_id,
                "stage": wire_stage,
                "detail": detail,
                "worker_id": worker_id,
            },
        )
