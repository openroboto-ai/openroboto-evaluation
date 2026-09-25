"""benchmark worker 子进程清理的回归测试。"""

import pathlib
import signal
import subprocess
import sys
import types
import unittest
from contextlib import ExitStack
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmark_worker import worker  # noqa: E402
from libero_eval.gpu_health import GpuHealth


class TestTerminate(unittest.TestCase):
    def test_terminates_entire_eval_process_group(self):
        process = mock.Mock(pid=123)

        with mock.patch.object(worker.os, "killpg") as killpg:
            worker._terminate(process)

        killpg.assert_called_once_with(123, signal.SIGTERM)
        process.wait.assert_called_once_with(timeout=30)
        process.terminate.assert_not_called()

    def test_unkillable_evaluation_does_not_block_worker_cleanup(self):
        process = mock.Mock(pid=123)
        process.wait.side_effect = subprocess.TimeoutExpired("evaluation", 30)
        with mock.patch.object(worker.os, "killpg") as killpg:
            worker._terminate(process)
        self.assertEqual(process.wait.call_args_list, [mock.call(timeout=30), mock.call(timeout=5)])
        self.assertEqual(killpg.call_args_list, [mock.call(123, signal.SIGTERM), mock.call(123, signal.SIGKILL)])

    def test_process_exit_race_is_harmless(self):
        process = mock.Mock(pid=123)
        with mock.patch.object(worker.os, "killpg", side_effect=ProcessLookupError):
            worker._terminate(process)


class TestGpuPause(unittest.TestCase):
    def test_recovers_before_starting_next_task(self):
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    worker,
                    "check_gpu_health",
                    side_effect=[
                        GpuHealth(False, "driver blocked"),
                        GpuHealth(True, "GPU 0: Test"),
                    ],
                )
            )
            wait = stack.enter_context(mock.patch.object(worker, "_wait_for_event"))
            stack.enter_context(mock.patch.object(worker, "_stopping", return_value=False))
            self.assertTrue(worker.wait_for_gpu_health())
        wait.assert_called_once_with(worker.stop_event, 60)

    def test_shutdown_interrupts_gpu_pause(self):
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(worker, "check_gpu_health", return_value=GpuHealth(False, "driver blocked"))
            )
            stack.enter_context(mock.patch.object(worker, "_wait_for_event"))
            stack.enter_context(mock.patch.object(worker, "_stopping", side_effect=[False, True]))
            self.assertFalse(worker.wait_for_gpu_health())

    def test_gpu_failure_interrupts_running_eval_before_eight_hour_timeout(self):
        import tempfile

        args = types.SimpleNamespace(
            benchmark="libero_pro",
            eval_config=None,
            num_trials=50,
            gpus="0",
            workers_per_gpu=1,
            server_impl="upstream",
            task_ids="",
            eval_timeout=28800,
        )
        task = {"task_id": "gpu-failure", "base_model": "pi0.5", "hf_commit": "a" * 40}
        process = mock.Mock(pid=123)
        process.poll.return_value = None
        with ExitStack() as stack:
            tmp = stack.enter_context(tempfile.TemporaryDirectory())
            stack.enter_context(mock.patch.object(worker.subprocess, "Popen", return_value=process))
            stack.enter_context(mock.patch.object(worker.time, "monotonic", side_effect=[0, 60, 60]))
            stack.enter_context(
                mock.patch.object(worker, "check_gpu_health", return_value=GpuHealth(False, "driver blocked"))
            )
            stack.enter_context(mock.patch.object(worker, "_stopping", return_value=False))
            terminate = stack.enter_context(mock.patch.object(worker, "_terminate"))
            with self.assertRaisesRegex(worker.EvalInfrastructureError, "GPU health check failed"):
                worker.run_evaluation(task, pathlib.Path(tmp), pathlib.Path(tmp), args)
        terminate.assert_called_once_with(process)


if __name__ == "__main__":
    unittest.main()
