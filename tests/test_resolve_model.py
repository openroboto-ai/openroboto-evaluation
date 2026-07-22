"""模型解析必须 pin 到具体 HF commit 的单元测试。

覆盖两处入口(评测缓存按 commit 分目录,重新提交同一 repo 不会命中旧缓存):
  - benchmark_worker/worker.py resolve_model:队列任务缺失/非法 hf_commit 在
    下载前即被拒绝(错误如实上报后端,miner 可见);
  - libero_eval/run_eval.py resolve_model:HF 引用必须携带完整 40 位 hex 的
    --commit-id 才能下载;本地目录不受影响。

运行:uv run python -m unittest discover tests
"""

import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "libero_eval"))  # run_eval 用平铺 import(from download import ...)

import run_eval  # noqa: E402
from benchmark_worker import worker  # noqa: E402

SHA = "a" * 40
SHA2 = "b" * 40
DL = pathlib.Path("/nonexistent/download-root")


def _task(repo: str = "u/r", commit: str | None = SHA):
    return {"task_id": "t1", "hf_repo_id": repo, "hf_commit": commit}


class TestWorkerResolveModel(unittest.TestCase):
    def test_download_dir_is_keyed_by_commit(self):
        ref, revision, local_dir = worker.resolve_model(_task(), DL, allow_local=False)
        self.assertEqual(ref, "u/r")
        self.assertEqual(revision, SHA)
        self.assertEqual(local_dir, (DL / f"u__r@{SHA[:12]}").resolve())

    def test_different_commits_never_share_a_dir(self):
        _, _, d1 = worker.resolve_model(_task(commit=SHA), DL, allow_local=False)
        _, _, d2 = worker.resolve_model(_task(commit=SHA2), DL, allow_local=False)
        self.assertNotEqual(d1, d2)

    def test_missing_commit_is_rejected(self):
        for commit in ("", "   ", None):
            with self.assertRaisesRegex(ValueError, "no hf_commit"):
                worker.resolve_model(_task(commit=commit), DL, allow_local=False)

    def test_non_sha_revision_is_rejected(self):
        # 分支名/tag/短哈希都会随内容漂移或有歧义,不能作为评测锚点。
        for commit in ("main", "v1.0", "A" * 40, "abc123", SHA + "0"):
            with self.assertRaisesRegex(ValueError, "invalid hf_commit"):
                worker.resolve_model(_task(commit=commit), DL, allow_local=False)

    def test_invalid_repo_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid hf_repo_id"):
            worker.resolve_model(_task(repo="not-a-repo"), DL, allow_local=False)

    def test_allow_local_accepts_existing_path_without_commit(self):
        local = pathlib.Path(__file__).resolve().parent
        ref, revision, d = worker.resolve_model({"hf_repo_id": str(local), "hf_commit": ""}, DL, allow_local=True)
        self.assertIsNone(ref)
        self.assertIsNone(revision)
        self.assertEqual(d, local)


class TestRunEvalResolveModel(unittest.TestCase):
    def test_local_openvla_oft_root_is_recognized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "model.safetensors.index.json").write_text("{}")
            # macOS exposes temporary directories through /var while
            # Path.resolve() canonicalizes the same path under /private/var.
            self.assertEqual(run_eval.resolve_model(str(root), root / "downloads"), root.resolve())

    def test_gpu_locks_reject_overlapping_processes(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {"LIBERO_EVAL_GPU_LOCK_DIR": td}):
                locks = run_eval.acquire_gpu_locks([7, 3])
            try:
                script = f"""
import os, sys
os.environ['LIBERO_EVAL_GPU_LOCK_DIR'] = {td!r}
sys.path.insert(0, {str(_ROOT / 'libero_eval')!r})
import run_eval
try:
    run_eval.acquire_gpu_locks([3])
except RuntimeError as exc:
    print(exc)
    raise SystemExit(0)
raise SystemExit(1)
"""
                result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("GPU 3 is locked by pid=", result.stdout)
            finally:
                for lock in locks:
                    lock.close()

    def test_gpu_locks_validate_selection(self):
        with self.assertRaisesRegex(ValueError, "At least one GPU"):
            run_eval.acquire_gpu_locks([])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            run_eval.acquire_gpu_locks([1, 1])

    def test_hf_repo_without_full_sha_is_rejected(self):
        for bad in (None, "", "main", "abc123", "A" * 40, "local"):
            with self.assertRaisesRegex(ValueError, "commit"):
                run_eval.resolve_model("u/r", DL, commit_id=bad)

    def test_hf_download_is_pinned_and_cached_per_commit(self):
        seen = {}

        def fake_download(repo_id, local_dir, *, revision=None, strategies=None, **_):
            seen["repo_id"], seen["revision"] = repo_id, revision
            (pathlib.Path(local_dir) / "params").mkdir(parents=True)
            return local_dir

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(run_eval, "download_model", fake_download):
                ckpt = run_eval.resolve_model("u/r", pathlib.Path(td), commit_id=SHA)
        self.assertEqual(seen["repo_id"], "u/r")
        self.assertEqual(seen["revision"], SHA)
        self.assertEqual(ckpt.name, f"u__r@{SHA[:12]}")

    def test_local_path_is_used_as_is(self):
        # 本地目录不下载,commit id 仅由调用方记录(worker 传任务的 hf_commit)。
        with tempfile.TemporaryDirectory() as td:
            (pathlib.Path(td) / "params").mkdir()
            ckpt = run_eval.resolve_model(td, DL, commit_id="local")
        self.assertEqual(ckpt, pathlib.Path(td).resolve())


if __name__ == "__main__":
    unittest.main()
