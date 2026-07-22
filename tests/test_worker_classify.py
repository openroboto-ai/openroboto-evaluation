"""benchmark_worker/worker.py 队列任务处置逻辑(classify_queued_task)的单元测试。

运行:uv run python -m unittest discover tests
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmark_worker.worker import _model_label, classify_queued_task  # noqa: E402

TASK = {"task_id": "t1", "hf_repo_id": "u/r", "hf_commit": "aaa"}
NEW_COMMIT = {"task_id": "t1", "hf_repo_id": "u/r", "hf_commit": "bbb"}
NEW_REPO = {"task_id": "t1", "hf_repo_id": "u/r2", "hf_commit": "aaa"}


def _entry(status, task=TASK, payload=None):
    e = {"status": status, "task": dict(task)}
    if payload is not None:
        e["payload"] = payload
    return e


class TestClassifyQueuedTask(unittest.TestCase):
    def test_unseen_task_is_evaluated(self):
        self.assertEqual(classify_queued_task(None, TASK), "evaluate")

    def test_in_progress_is_skipped_even_with_new_commit(self):
        # 本地正在排队/评测:先等它出结果,新 commit 下一轮再处理。
        for status in ("pending", "running"):
            self.assertEqual(classify_queued_task(_entry(status), TASK), "skip")
            self.assertEqual(classify_queued_task(_entry(status), NEW_COMMIT), "skip")

    def test_submitted_same_commit_resubmits_stored_result(self):
        # 已评完、后端也确认过,但队列里又出现 = 后端丢了结果 → 重发,不重评。
        entry = _entry("submitted", payload={"success": False})
        self.assertEqual(classify_queued_task(entry, TASK), "resubmit")

    def test_submitted_without_payload_is_skipped(self):
        self.assertEqual(classify_queued_task(_entry("submitted"), TASK), "skip")

    def test_new_commit_reevaluates(self):
        entry = _entry("submitted", payload={"success": False})
        self.assertEqual(classify_queued_task(entry, NEW_COMMIT), "evaluate")

    def test_new_repo_id_reevaluates(self):
        entry = _entry("submitted", payload={"success": False})
        self.assertEqual(classify_queued_task(entry, NEW_REPO), "evaluate")

    def test_done_pending_submit_same_commit_left_to_retry_path(self):
        # 补交由主循环的 done_pending_submit 路径负责,这里不掺和。
        entry = _entry("done_pending_submit", payload={"success": True})
        self.assertEqual(classify_queued_task(entry, TASK), "skip")

    def test_done_pending_submit_new_commit_drops_stale_result(self):
        # 换 commit 后旧结果作废:后端只关心当前提交。
        entry = _entry("done_pending_submit", payload={"success": True})
        self.assertEqual(classify_queued_task(entry, NEW_COMMIT), "evaluate")

    def test_abandoned_same_commit_is_skipped(self):
        self.assertEqual(classify_queued_task(_entry("abandoned"), TASK), "skip")

    def test_abandoned_new_commit_reevaluates(self):
        self.assertEqual(classify_queued_task(_entry("abandoned"), NEW_COMMIT), "evaluate")

    def test_profile_change_never_reuses_old_result(self):
        entry = _entry("submitted", payload={"success": True})
        entry["benchmark"] = "libero_pro"
        self.assertEqual(classify_queued_task(entry, TASK, "libero_pro_custom_1"), "evaluate")

    def test_old_libero_plus_protocol_result_is_reevaluated(self):
        entry = _entry("submitted", payload={"success": True})
        entry["benchmark"] = "libero_plus"
        self.assertEqual(classify_queued_task(entry, TASK, "libero_plus"), "evaluate")

        entry["protocol_revision"] = "libero_plus_official_v1"
        self.assertEqual(classify_queued_task(entry, TASK, "libero_plus"), "resubmit")


class TestModelLabel(unittest.TestCase):
    def test_repo_and_commit_truncated_to_12(self):
        task = {"hf_repo_id": "user/repo", "hf_commit": "0123456789abcdef" + "0" * 24}
        self.assertEqual(_model_label(task), "user/repo@0123456789ab")

    def test_missing_fields_fall_back_to_question_mark(self):
        self.assertEqual(_model_label({}), "?@?")
        self.assertEqual(_model_label(None), "?@?")
        self.assertEqual(_model_label({"hf_repo_id": "u/r"}), "u/r@?")


if __name__ == "__main__":
    unittest.main()
