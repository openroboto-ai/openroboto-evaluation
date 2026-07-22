"""
后端 Benchmark API 的最小 HTTP 客户端 | Minimal HTTP client for the backend benchmark API.

只覆盖 worker 需要的三个接口。队列与结果提交使用运行时提供的 worker API
key，并放在 X-API-Key 头中：

    GET  /api/pending-tasks               -> 待评测任务队列
    GET  /api/submission/{id}             -> 核对评分是否已经落库
    POST /api/v1/benchmark/task/{id}/score     -> 提交评测结果

仅用标准库(urllib),保持通信层零第三方依赖。
"""

import json
import urllib.error
import urllib.request

USER_AGENT = "validator-benchmark-worker/0.1"


class BackendError(Exception):
    """与后端通信失败(HTTP 或网络层)。

    `status` 为 HTTP 状态码,网络层错误时为 None。
    `permanent` 标记重试无意义的错误(4xx;但 401/403/429 除外)。
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status

    @property
    def permanent(self) -> bool:
        # 401/403 是本端 key 没配对(后端并没有否定结果本身),改对 key 重试
        # 即可成功;429 是限流。两者都不能作为丢弃已评测结果的依据。
        return self.status is not None and 400 <= self.status < 500 and self.status not in (401, 403, 429)


class BackendClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 30.0, submit_timeout_s: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.submit_timeout_s = submit_timeout_s

    def _request(self, method: str, path: str, body: dict | None = None, timeout_s: float | None = None) -> dict:
        # 后端未设 BACKEND_API_KEY 时鉴权整体关闭(本机/内网部署),此时不带头。
        # 显式标识客户端；Cloudflare 会以 1010 拒绝 Python-urllib 默认 UA。
        headers = {"User-Agent": USER_AGENT}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
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
                detail = e.read().decode(errors="replace")[:500]
            except OSError:
                detail = ""
            raise BackendError(f"{method} {path} -> HTTP {e.code}: {detail}", status=e.code) from e
        except urllib.error.URLError as e:
            raise BackendError(f"{method} {path} -> {e.reason}") from e
        except OSError as e:
            # Python 3.14 的 socket read timeout 会从 urllib 直接冒成
            # TimeoutError(OSError),而不是包装成 URLError。
            raise BackendError(f"{method} {path} -> {e}") from e

    def fetch_queue(self) -> list[dict]:
        """拉取后端待评测任务列表(可能包含本地已处理过的任务,由调用方去重)。

        响应体是 {"queue_size": N, "tasks": [...]} 信封(api_reference_zh.md 3.1),
        这里解开只返回任务列表。
        """
        data = self._request("GET", "/api/pending-tasks")
        return data.get("tasks", []) if isinstance(data, dict) else data

    def fetch_submission(self, task_id: str) -> dict:
        """读取任务详情,用于确认超时/5xx 的评分 POST 是否实际已经落库。"""
        return self._request("GET", f"/api/submission/{task_id}")

    def submit_score(self, task_id: str, payload: dict) -> dict:
        # 后端同步落库耗时可能明显长于普通 GET,单独留出充足超时。
        return self._request(
            "POST",
            f"/api/v1/benchmark/task/{task_id}/score",
            body=payload,
            timeout_s=self.submit_timeout_s,
        )
