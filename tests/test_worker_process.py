"""benchmark worker 子进程清理的回归测试。"""

import pathlib
import signal
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmark_worker import worker  # noqa: E402


class TestTerminate(unittest.TestCase):
    def test_terminates_entire_eval_process_group(self):
        process = mock.Mock(pid=123)

        with mock.patch.object(worker.os, "killpg") as killpg:
            worker._terminate(process)

        killpg.assert_called_once_with(123, signal.SIGTERM)
        process.wait.assert_called_once_with(timeout=30)
        process.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
