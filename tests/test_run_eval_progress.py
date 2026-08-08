"""Progress event contract between run_eval.py and benchmark_worker."""

import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "libero_eval"))

from run_eval import emit_progress_event  # noqa: E402


class TestEmitProgressEvent(unittest.TestCase):
    def test_appends_one_json_line_per_event(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            path = pathlib.Path(tmp_str) / "nested" / "progress.jsonl"

            emit_progress_event(path, {"suites_done": 0, "suites_total": 2})
            emit_progress_event(
                path,
                {
                    "suites_done": 1,
                    "suites_total": 2,
                    "last_completed_suite": "libero_goal",
                },
            )

            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(
                events,
                [
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
                ],
            )

    def test_none_path_is_disabled(self):
        emit_progress_event(None, {"suites_done": 0})


if __name__ == "__main__":
    unittest.main()
