import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))
from axis_recording import save_recording  # noqa: E402


class TestAxisRecording(unittest.TestCase):
    def test_frames_and_trace_have_verifiable_portable_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary) / "recordings"
            frames = [np.zeros((8, 8, 3), dtype=np.uint8), np.full((8, 8, 3), 255, dtype=np.uint8)]
            kwargs = dict(
                task_id=501,
                trial=0,
                frames=frames,
                steps=[{"step": 1, "checker": {"passed": True}}],
                control_period_s=0.2,
                metadata={"success": True},
            )
            result = save_recording(root, **kwargs)
            animation = pathlib.Path(temporary) / result["animation"]
            trace = pathlib.Path(temporary) / result["trace"]
            with Image.open(animation) as image:
                self.assertEqual(image.n_frames, 2)
                self.assertEqual(image.info["duration"], 200)
            self.assertEqual(json.loads(trace.read_text())["steps"][0]["checker"], {"passed": True})
            self.assertEqual(hashlib.sha256(trace.read_bytes()).hexdigest(), result["trace_sha256"])
            with self.assertRaises(FileExistsError):
                save_recording(root, **kwargs)

    def test_incomplete_recording_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "reset image"):
            save_recording(
                pathlib.Path(temporary),
                task_id=501,
                trial=0,
                frames=[np.zeros((8, 8, 3), dtype=np.uint8)],
                steps=[{}, {}],
                control_period_s=0.2,
                metadata={},
            )
