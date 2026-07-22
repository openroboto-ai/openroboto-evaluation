"""benchmark worker CLI 参数解析的回归测试。"""

import contextlib
import io
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmark_worker import worker  # noqa: E402


class TestParseArgs(unittest.TestCase):
    def test_help_renders_literal_percent_sign(self):
        output = io.StringIO()
        with mock.patch.object(sys, "argv", ["benchmark-worker", "--help"]):
            with contextlib.redirect_stdout(output), self.assertRaisesRegex(SystemExit, "0"):
                worker.parse_args()

        self.assertIn("~13%", output.getvalue())

    def test_rejects_invalid_gpu_wait_settings(self):
        for flag, value in (("--gpu-max-used-mib", "-1"), ("--gpu-wait-interval", "0")):
            with self.subTest(flag=flag):
                with mock.patch.object(sys, "argv", ["benchmark-worker", flag, value]):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(SystemExit, "2"):
                        worker.parse_args()


if __name__ == "__main__":
    unittest.main()
