"""benchmark_worker/state.py(StateStore)的单元测试。

运行:uv run python -m unittest discover tests
"""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmark_worker.state import StateStore  # noqa: E402


class TestStateStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = pathlib.Path(self.tmp.name) / "state.json"

    def test_missing_file_starts_fresh(self):
        store = StateStore(self.path)
        self.assertEqual(store.all_tasks(), {})

    def test_empty_file_starts_fresh(self):
        # 手动清空文件 = 要求全部重评,不应像坏 JSON 一样崩溃。
        self.path.write_text("")
        store = StateStore(self.path)
        self.assertEqual(store.all_tasks(), {})

    def test_corrupt_file_fails_with_actionable_error(self):
        self.path.write_text("{not json")
        with self.assertRaisesRegex(ValueError, "fix it or delete it"):
            StateStore(self.path)

    def test_update_persists_and_reloads(self):
        store = StateStore(self.path)
        store.update("t1", status="pending", task={"task_id": "t1"})
        store.update("t1", status="submitted")
        reloaded = StateStore(self.path)
        entry = reloaded.get("t1")
        assert entry is not None
        self.assertEqual(entry["status"], "submitted")
        self.assertEqual(entry["task"], {"task_id": "t1"})
        self.assertIn("updated_at", entry)

    def test_written_file_is_valid_json(self):
        store = StateStore(self.path)
        store.update("t1", status="pending")
        data = json.loads(self.path.read_text())
        self.assertIn("t1", data["tasks"])


if __name__ == "__main__":
    unittest.main()
