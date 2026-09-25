import pathlib
import subprocess
import sys
import tempfile
import unittest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "libero_eval"))

from download_hf_paths import manifest_paths  # noqa: E402


class TestManifestPaths(unittest.TestCase):
    def test_lists_prefix_and_excludes_training_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
            for name in ("ckpt/params/a", "ckpt/assets/stats", "ckpt/train_state/optimizer", "other/file"):
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
            revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()

            paths = manifest_paths(repo, revision, "ckpt", ("ckpt/train_state/",))

        self.assertEqual(paths, ["ckpt/assets/stats", "ckpt/params/a"])


if __name__ == "__main__":
    unittest.main()
