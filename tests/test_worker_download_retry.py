"""下载失败不产生失败报告:原地重试,耗尽后任务重新排队。

回归背景(2026-07-18):代理上游抖动导致三种下载策略同时 SSL 失败,worker
把 DownloadError 做成 success=false 报告提交,被后端"env 必须齐全"校验
400 拒收,任务被标 abandoned 永久卡死。下载失败是本端网络问题,不是对
模型的评测结论,不允许走上报路径。

运行:uv run python -m unittest discover tests
"""

import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmark_worker import worker  # noqa: E402
from benchmark_worker.state import StateStore  # noqa: E402
from libero_eval.download import DownloadError  # noqa: E402

TASK = {
    "task_id": "t1",
    "hf_repo_id": "u/r",
    "hf_commit": "a" * 40,
    "env_list": ["libero_spatial"],
}


def _args(tmp: pathlib.Path, retries: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        output_root=tmp / "runs",
        download_dir=tmp / "models",
        allow_local_model=False,
        no_init_randomization=True,
        download_strategies=[],
        download_retries=retries,
        gpus="0,1",
        gpu_max_used_mib=1024,
        gpu_wait_interval=30,
    )


class TestDownloadWithRetry(unittest.TestCase):
    def test_transient_failure_retried_until_success(self):
        calls = []

        def flaky(*a, **k):
            calls.append(1)
            if len(calls) < 3:
                raise DownloadError("boom")

        with mock.patch.object(worker, "download_model", side_effect=flaky):
            with mock.patch.object(worker.stop_event, "wait", return_value=False):
                worker.download_with_retry("u/r", "a" * 40, pathlib.Path("/x"), _args(pathlib.Path("/x"), retries=4))
        self.assertEqual(len(calls), 3)

    def test_exhausted_retries_raise(self):
        with mock.patch.object(worker, "download_model", side_effect=DownloadError("boom")) as dl:
            with mock.patch.object(worker.stop_event, "wait", return_value=False):
                with self.assertRaises(DownloadError):
                    worker.download_with_retry(
                        "u/r", "a" * 40, pathlib.Path("/x"), _args(pathlib.Path("/x"), retries=2)
                    )
        self.assertEqual(dl.call_count, 2)

    def test_shutdown_signal_interrupts_backoff(self):
        with mock.patch.object(worker, "download_model", side_effect=DownloadError("boom")):
            with mock.patch.object(worker.stop_event, "wait", return_value=True):
                with self.assertRaises(worker.EvalInterrupted):
                    worker.download_with_retry(
                        "u/r", "a" * 40, pathlib.Path("/x"), _args(pathlib.Path("/x"), retries=2)
                    )


class TestProcessTaskDownloadFailure(unittest.TestCase):
    def test_no_failure_report_and_task_requeued(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = pathlib.Path(tmp_str)
            store = StateStore(tmp / "state.json")
            store.update(
                "t1",
                status="abandoned",
                submit_attempts=21,
                submit_error="old submission failed",
                next_submit_at="2099-01-01T00:00:00+00:00",
                note="old result",
            )
            client = mock.Mock()
            with mock.patch.object(worker, "download_model", side_effect=DownloadError("all strategies failed")):
                worker.process_task(dict(TASK), client, store, _args(tmp, retries=1))
            entry = store.get("t1")
            self.assertIsNotNone(entry)
            self.assertEqual(entry["status"], "pending")
            self.assertIn("download failed", entry.get("note", ""))
            self.assertEqual(entry["submit_attempts"], 0)
            self.assertIsNone(entry["submit_error"])
            self.assertIsNone(entry["next_submit_at"])
            client.submit_score.assert_not_called()

class TestInfrastructureDetection(unittest.TestCase):
    def tearDown(self):
        worker.stop_event.clear()

    def test_detects_retryable_infrastructure_markers(self):
        self.assertTrue(worker._has_retryable_infrastructure_error("GPU reservation failed: GPU 4 is locked"))
        self.assertTrue(worker._has_retryable_infrastructure_error("XlaRuntimeError: RESOURCE_EXHAUSTED"))
        self.assertTrue(worker._has_retryable_infrastructure_error("CUDA out of memory"))
        self.assertTrue(
            worker._has_retryable_infrastructure_error(
                "mujoco.FatalError: Offscreen framebuffer is not complete, error 0x8cdd"
            )
        )
        self.assertFalse(worker._has_retryable_infrastructure_error("checkpoint shape mismatch"))

    def test_partial_summary_with_egl_failure_is_requeued(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            out_dir = pathlib.Path(tmp_str) / "run"
            (out_dir / "logs").mkdir(parents=True)
            (out_dir / "summary.json").write_text(
                '{"tasks":{"libero_spatial_task04":{"status":"failed"}}}'
            )
            (out_dir / "logs" / "libero_spatial_task04.log").write_text(
                "mujoco.FatalError: Offscreen framebuffer is not complete, error 0x8cdd\n"
            )
            args = types.SimpleNamespace(
                eval_config="pi05_libero",
                num_trials=10,
                gpus="0",
                workers_per_gpu=3,
                server_impl="upstream",
                task_ids="",
                eval_timeout=60,
            )
            proc = mock.Mock(returncode=2)
            proc.poll.return_value = 2

            with mock.patch.object(worker.subprocess, "Popen", return_value=proc):
                with self.assertRaisesRegex(worker.EvalInfrastructureError, "libero_spatial_task04"):
                    worker.run_evaluation(TASK, pathlib.Path(tmp_str) / "model", out_dir, args)

    def test_partial_summary_with_non_infra_failure_is_returned(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            out_dir = pathlib.Path(tmp_str) / "run"
            (out_dir / "logs").mkdir(parents=True)
            summary = {"tasks": {"libero_spatial_task04": {"status": "failed"}}}
            (out_dir / "summary.json").write_text(json.dumps(summary))
            (out_dir / "logs" / "libero_spatial_task04.log").write_text("invalid task configuration\n")
            args = types.SimpleNamespace(
                eval_config="pi05_libero",
                num_trials=10,
                gpus="0",
                workers_per_gpu=3,
                server_impl="upstream",
                task_ids="",
                eval_timeout=60,
            )
            proc = mock.Mock(returncode=2)
            proc.poll.return_value = 2

            with mock.patch.object(worker.subprocess, "Popen", return_value=proc):
                actual, error = worker.run_evaluation(TASK, pathlib.Path(tmp_str) / "model", out_dir, args)

            self.assertEqual(actual, summary)
            self.assertIn("failed after retries", error)


class TestEvaluationShutdownRace(unittest.TestCase):
    def tearDown(self):
        worker.stop_event.clear()

    def test_shutdown_wins_when_eval_process_already_exited(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            out_dir = pathlib.Path(tmp_str) / "run"
            out_dir.mkdir()
            args = types.SimpleNamespace(
                eval_config="pi05_libero",
                num_trials=10,
                gpus="0",
                workers_per_gpu=1,
                server_impl="upstream",
                task_ids="",
                eval_timeout=60,
            )
            proc = mock.Mock(returncode=1)
            proc.poll.return_value = 1
            worker.stop_event.set()

            with mock.patch.object(worker.subprocess, "Popen", return_value=proc):
                with self.assertRaises(worker.EvalInterrupted):
                    worker.run_evaluation(TASK, pathlib.Path(tmp_str) / "model", out_dir, args)


class TestSignalHandling(unittest.TestCase):
    def tearDown(self):
        worker.stop_event.clear()

    def test_handler_only_sets_stop_flag_without_reentrant_logging(self):
        worker.stop_event.clear()
        with mock.patch.object(worker.logger, "info", side_effect=RuntimeError("reentrant call")) as log:
            worker._handle_signal(worker.signal.SIGTERM, None)

        self.assertTrue(worker.stop_event.is_set())
        log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
