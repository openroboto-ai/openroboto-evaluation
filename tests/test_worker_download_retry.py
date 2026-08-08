"""下载/评测异常清缓存并重新排队，绝不发布不完整结果。

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
from contextlib import ExitStack
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

    def test_permanent_failure_is_not_retried(self):
        error = DownloadError("repository is malformed", permanent=True)
        with mock.patch.object(worker, "download_model", side_effect=error) as dl:
            with self.assertRaises(DownloadError) as ctx:
                worker.download_with_retry("u/r", "a" * 40, pathlib.Path("/x"), _args(pathlib.Path("/x"), retries=4))
        self.assertIs(ctx.exception, error)
        self.assertEqual(dl.call_count, 1)

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
            model_dir = tmp / "models" / f"u__r@{'a' * 12}"

            def fail_after_partial_download(*_args, **_kwargs):
                model_dir.mkdir(parents=True)
                (model_dir / "partial.safetensors").write_text("partial")
                raise DownloadError("all strategies failed")

            with mock.patch.object(worker, "download_model", side_effect=fail_after_partial_download):
                worker.process_task(dict(TASK), client, store, _args(tmp, retries=1))
            entry = store.get("t1")
            self.assertIsNotNone(entry)
            self.assertEqual(entry["status"], "pending")
            self.assertIn("download failed", entry.get("note", ""))
            self.assertEqual(entry["submit_attempts"], 0)
            self.assertIsNone(entry["submit_error"])
            self.assertIsNone(entry["next_submit_at"])
            self.assertIsNone(entry["out_dir"])
            self.assertFalse(model_dir.exists())
            self.assertEqual(list((tmp / "runs").iterdir()), [])
            client.submit_score.assert_not_called()

    def test_permanent_failure_also_clears_cache_and_requeues_without_report(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = pathlib.Path(tmp_str)
            store = StateStore(tmp / "state.json")
            client = mock.Mock()
            error = DownloadError(
                "model artifact integrity mismatch for u/r@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa: "
                "params/weights.bin expected 212074496 bytes but downloaded 891289600 bytes",
                permanent=True,
            )
            model_dir = tmp / "models" / f"u__r@{'a' * 12}"

            def fail_after_corrupt_download(*_args, **_kwargs):
                model_dir.mkdir(parents=True)
                (model_dir / "params").mkdir()
                (model_dir / "params" / "weights.bin").write_text("corrupt")
                raise error

            with mock.patch.object(worker, "download_model", side_effect=fail_after_corrupt_download) as dl:
                worker.process_task(dict(TASK), client, store, _args(tmp, retries=4))

            self.assertEqual(dl.call_count, 1)
            client.submit_score.assert_not_called()
            entry = store.get("t1")
            self.assertEqual(entry["status"], "pending")
            self.assertIn("download failed (permanent)", entry["note"])
            self.assertFalse(model_dir.exists())
            self.assertEqual(list((tmp / "runs").iterdir()), [])


class TestProcessTaskEvaluationFailure(unittest.TestCase):
    def test_partial_custom_benchmark_cache_is_deleted_and_full_run_requeued(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = pathlib.Path(tmp_str)
            args = _args(tmp, retries=1)
            args.benchmark = "libero_pro_custom_1"
            args.worker_id = "validator-host-a"
            store = StateStore(tmp / "state.json")
            client = mock.Mock()
            model_dir = tmp / "models" / f"u__r@{'a' * 12}"
            model_dir.mkdir(parents=True)
            (model_dir / "model.safetensors").write_text("complete model")

            def fail_after_ten_tasks(_task, _model_dir, out_dir, _args, **kwargs):
                self.assertNotIn("resume", kwargs)
                results = out_dir / "results"
                results.mkdir()
                for index in range(10):
                    (results / f"task_{index}.json").write_text('{"status":"ok"}')
                raise worker.EvalInfrastructureError("summary contains 10/160 required tasks")

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        worker,
                        "resolve_model",
                        return_value=("u/r", "a" * 40, model_dir),
                    )
                )
                stack.enter_context(mock.patch.object(worker, "download_with_retry"))
                stack.enter_context(mock.patch.object(worker, "run_evaluation", side_effect=fail_after_ten_tasks))
                worker.process_task(dict(TASK), client, store, args)

            entry = store.get("t1")
            self.assertEqual(entry["status"], "pending")
            self.assertIsNone(entry["payload"])
            self.assertIsNone(entry["out_dir"])
            self.assertFalse(entry["resume_evaluation"])
            self.assertIn("10/160", entry["note"])
            self.assertTrue(model_dir.exists(), "verified model download should be kept after an eval-only failure")
            self.assertEqual(list((tmp / "runs").iterdir()), [])
            client.submit_score.assert_not_called()

    def test_unexpected_evaluator_exception_also_deletes_partial_cache(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = pathlib.Path(tmp_str)
            args = _args(tmp, retries=1)
            args.benchmark = "libero_pro_custom_1"
            store = StateStore(tmp / "state.json")
            client = mock.Mock()

            def crash_after_partial_output(_task, _model_dir, out_dir, _args, **_kwargs):
                (out_dir / "results").mkdir()
                (out_dir / "results" / "task_00.json").write_text('{"status":"ok"}')
                raise RuntimeError("evaluator crashed")

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        worker,
                        "resolve_model",
                        return_value=(None, None, tmp / "local-model"),
                    )
                )
                stack.enter_context(mock.patch.object(worker, "run_evaluation", side_effect=crash_after_partial_output))
                with self.assertLogs("benchmark_worker", level="ERROR"):
                    worker.process_task(dict(TASK), client, store, args)

            entry = store.get("t1")
            self.assertEqual(entry["status"], "pending")
            self.assertIn("unexpected evaluating failure", entry["note"])
            self.assertEqual(list((tmp / "runs").iterdir()), [])
            client.submit_score.assert_not_called()

    def test_old_pending_output_is_cleaned_before_new_attempt(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = pathlib.Path(tmp_str)
            args = _args(tmp, retries=1)
            args.benchmark = "libero_pro_custom_1"
            args.worker_id = "validator-host-a"
            old_out = tmp / "runs" / "old_partial_t1"
            old_out.mkdir(parents=True)
            (old_out / "completed_task.json").write_text('{"status":"ok"}')
            store = StateStore(tmp / "state.json")
            store.update(
                "t1",
                status="pending",
                task=TASK,
                out_dir=str(old_out),
                resume_evaluation=False,
            )
            client = mock.Mock()

            def reject_model(_task, _model_dir, out_dir, _args, **kwargs):
                self.assertFalse(old_out.exists())
                self.assertNotEqual(out_dir, old_out)
                self.assertNotIn("resume", kwargs)
                return None, "model rejected by pre-eval check"

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        worker,
                        "resolve_model",
                        return_value=(None, None, tmp / "local-model"),
                    )
                )
                stack.enter_context(mock.patch.object(worker, "run_evaluation", side_effect=reject_model))
                stack.enter_context(mock.patch.object(worker, "build_score_payload", return_value={"success": False}))
                stack.enter_context(mock.patch.object(worker, "try_submit", return_value=True))
                worker.process_task(dict(TASK), client, store, args)

            self.assertFalse(old_out.exists())


class TestCacheCleanupSafety(unittest.TestCase):
    def test_historical_incomplete_success_payload_is_never_submitted(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = pathlib.Path(tmp_str)
            old_out = tmp / "runs" / "partial_t1"
            old_out.mkdir(parents=True)
            store = StateStore(tmp / "state.json")
            payload = {
                "benchmark": "libero_pro_custom_1",
                "success": True,
                "hf_repo_id": "u/r",
                "hf_commit": "a" * 40,
                "error": "63 task(s) failed after retries",
                "per_task_scores": [
                    {"task_id": f"task-{index}", "success_rate": 0.5, "trials": 50} for index in range(97)
                ],
                "env_scores": [{"env_name": "libero_spatial", "score": 0.5, "samples": 500}],
            }
            store.update(
                "t1",
                status="done_pending_submit",
                task=TASK,
                out_dir=str(old_out),
                payload=payload,
            )
            client = mock.Mock()

            self.assertFalse(worker.try_submit(client, store, "t1", payload))

            client.submit_score.assert_not_called()
            entry = store.get("t1")
            self.assertEqual(entry["status"], "pending")
            self.assertIsNone(entry["payload"])
            self.assertIsNone(entry["out_dir"])
            self.assertEqual(entry["cleanup_evaluation_dir"], str(old_out))

    def test_dirty_cache_outside_configured_root_blocks_retry(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = pathlib.Path(tmp_str)
            args = _args(tmp, retries=1)
            args.benchmark = "libero_pro_custom_1"
            outside = tmp / "outside" / "partial"
            outside.mkdir(parents=True)
            store = StateStore(tmp / "state.json")
            store.update(
                "t1",
                status="pending",
                task=TASK,
                cleanup_evaluation_dir=str(outside),
            )
            client = mock.Mock()

            with mock.patch.object(worker, "run_evaluation") as run:
                worker.process_task(dict(TASK), client, store, args)

            run.assert_not_called()
            entry = store.get("t1")
            self.assertEqual(entry["status"], "pending")
            self.assertEqual(entry["cleanup_evaluation_dir"], str(outside))
            self.assertIn("retry is blocked until clean", entry["note"])
            self.assertTrue(outside.exists())
            client.submit_score.assert_not_called()


class TestProgressEvents(unittest.TestCase):
    def test_process_task_reports_three_stages_in_order(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = pathlib.Path(tmp_str)
            args = _args(tmp, retries=1)
            args.benchmark = "libero_pro_custom_1"
            args.worker_id = "validator-host-a"
            store = StateStore(tmp / "state.json")
            client = mock.Mock()

            def fake_run(*_args, **kwargs):
                kwargs["progress_callback"](
                    "evaluating",
                    {"suites_done": 0, "suites_total": 16},
                )
                return {"tasks": {}}, ""

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        worker,
                        "resolve_model",
                        return_value=("u/r", "a" * 40, tmp / "model"),
                    )
                )
                stack.enter_context(mock.patch.object(worker, "download_with_retry"))
                stack.enter_context(mock.patch.object(worker, "run_evaluation", side_effect=fake_run))
                stack.enter_context(mock.patch.object(worker, "build_score_payload", return_value={"success": True}))
                stack.enter_context(mock.patch.object(worker, "try_submit", return_value=True))
                worker.process_task(dict(TASK), client, store, args)

            self.assertEqual(
                client.report_progress.call_args_list,
                [
                    mock.call("t1", "downloading", {}, "validator-host-a"),
                    mock.call("t1", "prechecking", {}, "validator-host-a"),
                    mock.call(
                        "t1",
                        "evaluating",
                        {"suites_done": 0, "suites_total": 16},
                        "validator-host-a",
                    ),
                ],
            )

    def test_forwards_every_complete_jsonl_event(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            path = pathlib.Path(tmp_str) / "progress.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(event)
                    for event in [
                        {
                            "stage": "evaluating",
                            "detail": {"suites_done": 0, "suites_total": 2},
                        },
                        {
                            "stage": "evaluating",
                            "detail": {
                                "suites_done": 1,
                                "suites_total": 2,
                                "last_completed_suite": "libero_goal",
                            },
                        },
                    ]
                )
                + "\n"
            )
            forwarded = []

            offset = worker._forward_progress_events(path, 0, lambda stage, detail: forwarded.append((stage, detail)))

            self.assertEqual(offset, path.stat().st_size)
            self.assertEqual(
                forwarded,
                [
                    ("evaluating", {"suites_done": 0, "suites_total": 2}),
                    (
                        "evaluating",
                        {
                            "suites_done": 1,
                            "suites_total": 2,
                            "last_completed_suite": "libero_goal",
                        },
                    ),
                ],
            )

    def test_keeps_offset_before_incomplete_event(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            path = pathlib.Path(tmp_str) / "progress.jsonl"
            path.write_text('{"stage":"evaluating"')
            forwarded = []

            offset = worker._forward_progress_events(path, 0, lambda stage, detail: forwarded.append((stage, detail)))

            self.assertEqual(offset, 0)
            self.assertEqual(forwarded, [])

    def test_progress_http_failure_is_non_fatal(self):
        client = mock.Mock()
        client.report_progress.side_effect = worker.BackendError("temporarily unavailable", status=503)

        with self.assertLogs("benchmark_worker", level="WARNING"):
            success = worker._report_progress(client, "task_1", "downloading", {}, "host-a")

        self.assertFalse(success)


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
            (out_dir / "summary.json").write_text('{"tasks":{"libero_spatial_task04":{"status":"failed"}}}')
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

    def test_partial_summary_with_websocket_handshake_timeout_is_requeued(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            out_dir = pathlib.Path(tmp_str) / "run"
            (out_dir / "logs").mkdir(parents=True)
            summary = {"tasks": {"libero_spatial_task04": {"status": "failed"}}}
            (out_dir / "summary.json").write_text(json.dumps(summary))
            (out_dir / "logs" / "libero_spatial_task04.log").write_text("TimeoutError: timed out during handshake\n")
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
                with self.assertRaisesRegex(worker.EvalInfrastructureError, "refusing to publish a partial score"):
                    worker.run_evaluation(TASK, pathlib.Path(tmp_str) / "model", out_dir, args)


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
