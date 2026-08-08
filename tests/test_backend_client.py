"""benchmark_worker 提交错误分类(BackendError.permanent / try_submit)的单元测试。

回归背景:后端给 POST /api/v1/benchmark/task/{id}/score 加了 admin key 鉴权后,
一次 401 曾被当作"永久拒绝"把已评完的结果标成 abandoned。401/403 是本端
key 配置问题,必须保持可重试,不能丢结果。

运行:uv run python -m unittest discover tests
"""

import datetime
import json
import os
import pathlib
import queue
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmark_worker.backend_client import (  # noqa: E402
    DEFAULT_QUEUE_PATH,
    BackendClient,
    BackendError,
)
from benchmark_worker.state import StateStore  # noqa: E402
from benchmark_worker import worker  # noqa: E402
from benchmark_worker.worker import submit_retry_due, try_submit  # noqa: E402


class TestBackendErrorPermanent(unittest.TestCase):
    def test_validation_4xx_is_permanent(self):
        for status in (400, 404, 413, 422):
            self.assertTrue(BackendError("x", status=status).permanent, status)

    def test_auth_and_rate_limit_are_retryable(self):
        # 401/403:key 没配对,改对配置重试即可;429:限流。
        for status in (401, 403, 429):
            self.assertFalse(BackendError("x", status=status).permanent, status)

    def test_server_and_network_errors_are_retryable(self):
        for status in (500, 502, 503, None):
            self.assertFalse(BackendError("x", status=status).permanent, status)


class TestFetchQueue(unittest.TestCase):
    def test_sends_explicit_user_agent(self):
        client = BackendClient("http://x", "public", "admin")
        response = mock.MagicMock()
        response.read.return_value = b"{}"
        context = mock.MagicMock()
        context.__enter__.return_value = response
        with mock.patch("urllib.request.urlopen", return_value=context) as urlopen:
            client._request("GET", "/probe", api_key="public")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("User-agent"), "validator-benchmark-worker/0.1")
        self.assertEqual(request.get_header("X-api-key"), "public")

    def test_bare_timeout_is_wrapped_as_retryable_backend_error(self):
        client = BackendClient("http://x", "public", "admin")
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(BackendError) as ctx:
                client._request("GET", "/probe", api_key="public")

        self.assertIsNone(ctx.exception.status)
        self.assertFalse(ctx.exception.permanent)
        self.assertIn("timed out", str(ctx.exception))

    def test_submit_uses_extended_timeout(self):
        client = BackendClient("http://x", "public", "admin", timeout_s=3, submit_timeout_s=123)
        with mock.patch.object(client, "_request", return_value={"ok": True}) as request:
            client.submit_score("t1", {"success": True})

        request.assert_called_once_with(
            "POST",
            "/api/v1/benchmark/task/t1/score",
            api_key="admin",
            body={"success": True},
            timeout_s=123,
        )

    def test_progress_uses_admin_key_and_maps_evaluating_to_running(self):
        client = BackendClient("http://x", "public", "admin")
        detail = {"suites_done": 5, "suites_total": 16}
        with mock.patch.object(client, "_request", return_value={"success": True}) as request:
            actual = client.report_progress("task_1", "evaluating", detail, "validator-01")

        self.assertEqual(actual, {"success": True})
        request.assert_called_once_with(
            "POST",
            "/api/benchmark-progress",
            api_key="admin",
            body={
                "task_id": "task_1",
                "stage": "running",
                "detail": detail,
                "worker_id": "validator-01",
            },
        )

    def test_progress_rejects_unknown_stage_before_request(self):
        client = BackendClient("http://x", "public", "admin")
        with mock.patch.object(client, "_request") as request:
            with self.assertRaisesRegex(ValueError, "invalid benchmark progress stage"):
                client.report_progress("task_1", "unknown", {}, "validator-01")
        request.assert_not_called()

    def test_fetch_submission_uses_public_key(self):
        client = BackendClient("http://x", "public", "admin")
        with mock.patch.object(client, "_request", return_value={"task_id": "t1"}) as request:
            self.assertEqual(client.fetch_submission("t1"), {"task_id": "t1"})

        request.assert_called_once_with("GET", "/api/submission/t1", api_key="public")

    def test_benchmark_queue_uses_public_key_and_unwraps_envelope(self):
        client = BackendClient("http://x", "public", "admin")
        envelope = {"queue_size": 1, "tasks": [{"task_id": "t1"}]}
        with mock.patch.object(client, "_request", return_value=envelope) as req:
            self.assertEqual(client.fetch_queue(), [{"task_id": "t1"}])
            req.assert_called_once_with("GET", DEFAULT_QUEUE_PATH, api_key="public")

    def test_queue_error_propagates(self):
        client = BackendClient("http://x", "public", "admin")
        with mock.patch.object(
            client,
            "_request",
            side_effect=BackendError("unauthorized", status=401),
        ) as req:
            with self.assertRaises(BackendError):
                client.fetch_queue()

        req.assert_called_once_with("GET", DEFAULT_QUEUE_PATH, api_key="public")

    def test_bare_list_response_tolerated(self):
        client = BackendClient("http://x", "public", "admin")
        with mock.patch.object(client, "_request", return_value=[{"task_id": "t1"}]):
            self.assertEqual(client.fetch_queue(), [{"task_id": "t1"}])

    def test_one_key_constructor_remains_compatible(self):
        client = BackendClient("http://x", "legacy")
        self.assertEqual(client.public_api_key, "legacy")
        self.assertEqual(client.admin_api_key, "legacy")


class TestWorkerApiKeyArgs(unittest.TestCase):
    def _parse(self, *extra: str):
        argv = [
            "worker.py",
            "--backend-url",
            "https://backend.example",
            "--benchmark",
            "libero_pro_custom_1",
            "--num-trials",
            "50",
            "--download-strategies",
            "hfd-mirror",
            *extra,
        ]
        with mock.patch.object(sys, "argv", argv):
            return worker.parse_args()

    def test_accepts_split_cli_keys(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            args = self._parse("--public-api-key", "public", "--admin-api-key", "admin")

        self.assertEqual(args.public_api_key, "public")
        self.assertEqual(args.admin_api_key, "admin")
        self.assertEqual(args.queue_path, "/api/v1/benchmark/queue")

    def test_accepts_split_environment_keys(self):
        env = {
            "BACKEND_PUBLIC_API_KEY": "public",
            "BACKEND_ADMIN_API_KEY": "admin",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            args = self._parse()

        self.assertEqual(args.public_api_key, "public")
        self.assertEqual(args.admin_api_key, "admin")

    def test_legacy_key_populates_both_roles(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            args = self._parse("--api-key", "legacy")

        self.assertEqual(args.public_api_key, "legacy")
        self.assertEqual(args.admin_api_key, "legacy")

    def test_missing_either_key_fails_fast(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                self._parse("--public-api-key", "public")
            with self.assertRaises(SystemExit):
                self._parse("--admin-api-key", "admin")


class _FailingClient:
    def __init__(self, status):
        self.status = status
        self.fetch_calls = 0

    def submit_score(self, task_id, payload):
        raise BackendError(f"POST .../{task_id}/score -> HTTP {self.status}", status=self.status)

    def fetch_submission(self, task_id):
        self.fetch_calls += 1
        raise BackendError(f"GET .../{task_id} -> HTTP 503", status=503)


def _score_payload(total_score=0.8085):
    return {
        "benchmark": "libero_pro_custom_1",
        "success": True,
        "miner_hotkey": "hotkey-1",
        "hf_repo_id": "user/model",
        "hf_commit": "a" * 40,
        "round_num": 7,
        "total_score": total_score,
        "env_scores": [
            {
                "base_suite": "libero_spatial",
                "perturbation": "swap",
                "env_name": "libero_spatial_swap",
                "score": 0.97,
                "samples": 500,
            },
            {
                "base_suite": "libero_object",
                "perturbation": "swap",
                "env_name": "libero_object_swap",
                "score": 0.647,
                "samples": 500,
            },
        ],
        "error": "",
        "expected_trials_per_task": 50,
        "per_task_scores": [{"task_id": f"task-{i}", "success_rate": 0.5, "trials": 50} for i in range(160)],
    }


def _remote_submission(payload, result_as_json=False, include_benchmark=True):
    result = {
        "success": payload["success"],
        "total_score": payload["total_score"],
        "env_scores": list(reversed(payload["env_scores"])),
    }
    if include_benchmark:
        result["benchmark"] = payload["benchmark"]
    return {
        "task_id": "t1",
        "miner_hotkey": payload["miner_hotkey"],
        "hf_repo_id": payload["hf_repo_id"],
        "hf_commit": payload["hf_commit"],
        "round_num": payload["round_num"],
        "status": "done",
        "result": json.dumps(result) if result_as_json else result,
    }


class _PersistedFailingClient(_FailingClient):
    def __init__(self, remote):
        super().__init__(502)
        self.remote = remote

    def fetch_submission(self, task_id):
        return self.remote


class TestTrySubmitAuthFailure(unittest.TestCase):
    def _store(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return StateStore(pathlib.Path(tmp.name) / "state.json")

    def test_401_keeps_result_for_retry(self):
        store = self._store()
        store.update("t1", status="done_pending_submit", payload={"success": True})
        done = try_submit(_FailingClient(401), store, "t1", {"success": True, "error": ""})
        self.assertFalse(done)  # 可重试,不是终态
        self.assertEqual(store.get("t1")["status"], "done_pending_submit")  # 结果保留
        self.assertEqual(store.get("t1")["submit_attempts"], 1)
        self.assertFalse(submit_retry_due(store.get("t1")))

    def test_400_marks_abandoned(self):
        store = self._store()
        store.update("t1", status="done_pending_submit", payload={"success": True})
        client = _FailingClient(400)
        done = try_submit(client, store, "t1", {"success": True, "error": ""})
        self.assertTrue(done)
        self.assertEqual(store.get("t1")["status"], "abandoned")
        self.assertEqual(client.fetch_calls, 0)

    def test_successful_submit_omits_task_details_but_preserves_stored_payload(self):
        store = self._store()
        payload = {
            "success": True,
            "total_score": 0.5,
            "error": "",
            "per_task_scores": [{"task_id": "libero_spatial_00", "success_rate": 0.5, "trials": 10}],
        }
        store.update("t1", status="done_pending_submit", payload=payload)
        client = mock.Mock()

        done = try_submit(client, store, "t1", payload)

        self.assertTrue(done)
        submitted_payload = client.submit_score.call_args.args[1]
        self.assertEqual(submitted_payload["per_task_scores"], [])
        self.assertEqual(store.get("t1")["payload"]["per_task_scores"], payload["per_task_scores"])
        self.assertEqual(store.get("t1")["status"], "submitted")
        self.assertEqual(store.get("t1")["submit_attempts"], 0)
        self.assertIsNone(store.get("t1")["next_submit_at"])

    def test_submit_logs_exact_http_payload(self):
        store = self._store()
        payload = {
            "success": True,
            "total_score": 0.5,
            "error": "",
            "per_task_scores": [{"task_id": "libero_spatial_00", "success_rate": 0.5, "trials": 10}],
        }
        client = mock.Mock()

        with self.assertLogs("benchmark_worker", level="INFO") as captured:
            try_submit(client, store, "task-1", payload)

        submitted = client.submit_score.call_args.args[1]
        expected_json = json.dumps(submitted, indent=2, ensure_ascii=False, sort_keys=True)
        output = "\n".join(captured.output)
        self.assertIn("POST /api/v1/benchmark/task/task-1/score payload:", output)
        self.assertIn(expected_json, output)
        self.assertEqual(submitted["per_task_scores"], [])

    def test_retry_backoff_persists_and_grows(self):
        store = self._store()
        payload = {"success": True, "error": ""}
        store.update("t1", status="done_pending_submit", payload=payload)
        client = _FailingClient(502)

        self.assertFalse(try_submit(client, store, "t1", payload))
        first = store.get("t1")
        self.assertEqual(first["submit_attempts"], 1)

        self.assertFalse(try_submit(client, store, "t1", payload))
        second = store.get("t1")
        self.assertEqual(second["submit_attempts"], 2)
        self.assertGreater(second["next_submit_at"], first["next_submit_at"])

    def test_502_with_exact_remote_result_is_treated_as_submitted(self):
        store = self._store()
        payload = _score_payload()
        store.update("t1", status="done_pending_submit", payload=payload, submit_attempts=3)
        client = _PersistedFailingClient(_remote_submission(payload))

        self.assertTrue(try_submit(client, store, "t1", payload))

        entry = store.get("t1")
        self.assertEqual(entry["status"], "submitted")
        self.assertEqual(entry["submit_attempts"], 0)
        self.assertIsNone(entry["submit_error"])
        self.assertIsNone(entry["next_submit_at"])
        self.assertIn("remote score verified", entry["note"])

    def test_502_with_json_encoded_remote_result_is_treated_as_submitted(self):
        store = self._store()
        payload = _score_payload()
        store.update("t1", status="done_pending_submit", payload=payload)
        client = _PersistedFailingClient(_remote_submission(payload, result_as_json=True))

        self.assertTrue(try_submit(client, store, "t1", payload))
        self.assertEqual(store.get("t1")["status"], "submitted")

    def test_502_with_legacy_remote_result_without_benchmark_is_treated_as_submitted(self):
        store = self._store()
        payload = _score_payload()
        store.update("t1", status="done_pending_submit", payload=payload)
        remote = _remote_submission(payload, include_benchmark=False)
        remote["hotkey"] = remote.pop("miner_hotkey")
        client = _PersistedFailingClient(remote)

        self.assertTrue(try_submit(client, store, "t1", payload))
        self.assertEqual(store.get("t1")["status"], "submitted")

    def test_502_with_conflicting_remote_benchmark_keeps_retrying(self):
        store = self._store()
        payload = _score_payload()
        remote = _remote_submission(payload)
        remote["result"]["benchmark"] = "libero_pro"
        store.update("t1", status="done_pending_submit", payload=payload)

        self.assertFalse(try_submit(_PersistedFailingClient(remote), store, "t1", payload))
        self.assertEqual(store.get("t1")["status"], "done_pending_submit")

    def test_502_with_mismatched_remote_total_keeps_retrying(self):
        store = self._store()
        payload = _score_payload()
        remote = _remote_submission(payload)
        remote["result"]["total_score"] = 0.1
        store.update("t1", status="done_pending_submit", payload=payload)

        self.assertFalse(try_submit(_PersistedFailingClient(remote), store, "t1", payload))
        self.assertEqual(store.get("t1")["status"], "done_pending_submit")

    def test_502_with_mismatched_remote_result_keeps_retrying(self):
        store = self._store()
        payload = _score_payload()
        remote = _remote_submission(payload)
        remote["result"]["env_scores"][0] = dict(remote["result"]["env_scores"][0], score=0.1)
        store.update("t1", status="done_pending_submit", payload=payload)
        client = _PersistedFailingClient(remote)

        self.assertFalse(try_submit(client, store, "t1", payload))

        entry = store.get("t1")
        self.assertEqual(entry["status"], "done_pending_submit")
        self.assertEqual(entry["submit_attempts"], 1)
        self.assertIsNotNone(entry["next_submit_at"])

    def test_retry_due_handles_future_past_and_invalid_values(self):
        now = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        self.assertFalse(submit_retry_due({"next_submit_at": "2026-01-01T00:01:00+00:00"}, now))
        self.assertTrue(submit_retry_due({"next_submit_at": "2025-12-31T23:59:00+00:00"}, now))
        self.assertTrue(submit_retry_due({"next_submit_at": "invalid"}, now))


class TestWorkerCrashState(unittest.TestCase):
    def test_submit_crash_preserves_completed_payload(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = StateStore(pathlib.Path(tmp.name) / "state.json")
        task = {"task_id": "t1", "hf_repo_id": "u/r", "hf_commit": "a" * 40}

        def crash_after_payload(_task, _client, state, _args):
            state.update("t1", status="done_pending_submit", payload={"success": True})
            raise TimeoutError("timed out")

        local_q = queue.Queue()
        local_q.put(task)
        local_q.put(None)
        worker.stop_event.clear()
        with mock.patch.object(worker, "process_task", side_effect=crash_after_payload):
            with self.assertLogs("benchmark_worker", level="ERROR"):
                worker.worker_loop(local_q, mock.Mock(), store, mock.Mock())

        self.assertEqual(store.get("t1")["status"], "done_pending_submit")


if __name__ == "__main__":
    unittest.main()
