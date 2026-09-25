"""Benchmark worker file logging isolation and daily file switching tests."""

import contextlib
import datetime
import io
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmark_worker import worker  # noqa: E402


class TestWorkerLogging(unittest.TestCase):
    def tearDown(self):
        for handler in worker.logger.handlers[:]:
            worker.logger.removeHandler(handler)
            handler.close()

    def test_backend_namespace_separates_dev_and_production(self):
        self.assertEqual(worker._backend_log_namespace("https://dev.example.invalid"), "dev.example.invalid")
        self.assertEqual(
            worker._backend_log_namespace("https://production.example.invalid"), "production.example.invalid"
        )

    def test_logger_writes_to_backend_specific_directory(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            root = pathlib.Path(tmp_str)
            day = datetime.date(2026, 8, 9)
            dev_path = worker._setup_logger("https://dev.example.invalid", root, date_provider=lambda: day)
            worker.logger.info("dev-only-record")
            for handler in worker.logger.handlers:
                handler.flush()

            prod_path = worker._setup_logger("https://production.example.invalid", root, date_provider=lambda: day)
            worker.logger.info("production-only-record")
            for handler in worker.logger.handlers:
                handler.flush()

            self.assertEqual(dev_path, root / "dev.example.invalid" / "benchmark_worker-2026-08-09.log")
            self.assertEqual(prod_path, root / "production.example.invalid" / "benchmark_worker-2026-08-09.log")
            self.assertIn("dev-only-record", dev_path.read_text())
            self.assertNotIn("production-only-record", dev_path.read_text())
            self.assertIn("production-only-record", prod_path.read_text())
            self.assertNotIn("dev-only-record", prod_path.read_text())

    def test_file_handler_switches_to_new_dated_file(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            root = pathlib.Path(tmp_str)
            current_day = [datetime.date(2026, 8, 9)]
            first_path = worker._setup_logger("https://dev.example.invalid", root, date_provider=lambda: current_day[0])
            worker.logger.info("before-midnight")

            current_day[0] = datetime.date(2026, 8, 10)
            worker.logger.info("after-midnight")
            for handler in worker.logger.handlers:
                handler.flush()

            second_path = root / "dev.example.invalid" / "benchmark_worker-2026-08-10.log"
            self.assertIn("before-midnight", first_path.read_text())
            self.assertNotIn("after-midnight", first_path.read_text())
            self.assertIn("after-midnight", second_path.read_text())

    def test_file_handler_removes_logs_older_than_retention_window(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            log_dir = pathlib.Path(tmp_str) / "dev.example.invalid"
            log_dir.mkdir(parents=True)
            expired = log_dir / "benchmark_worker-2026-07-10.log"
            kept = log_dir / "benchmark_worker-2026-07-11.log"
            expired.write_text("expired")
            kept.write_text("kept")

            worker._setup_logger(
                "https://dev.example.invalid",
                pathlib.Path(tmp_str),
                date_provider=lambda: datetime.date(2026, 8, 9),
            )

            self.assertFalse(expired.exists())
            self.assertTrue(kept.exists())

    def test_process_stderr_is_persisted_in_worker_log(self):
        with tempfile.TemporaryDirectory() as tmp_str, contextlib.redirect_stderr(io.StringIO()):
            root = pathlib.Path(tmp_str)
            day = datetime.date(2026, 8, 21)
            log_path = worker._setup_logger("https://dev.example.invalid", root, date_provider=lambda: day)
            redirected = worker._install_stderr_logging()

            sys.stderr.write("Traceback (most recent call last):\n")
            sys.stderr.write("http.client.IncompleteRead: response truncated\n")
            redirected.flush()

            log_text = log_path.read_text()
            self.assertIn("stderr: Traceback (most recent call last):", log_text)
            self.assertIn("stderr: http.client.IncompleteRead: response truncated", log_text)


if __name__ == "__main__":
    unittest.main()
