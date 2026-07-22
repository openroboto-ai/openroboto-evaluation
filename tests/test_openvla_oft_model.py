import json
import pathlib
import tempfile
import unittest

from libero_eval.check_model import check_openvla_oft_model, detect_model_family


class OpenVlaOftModelCheckTest(unittest.TestCase):
    def make_checkpoint(self, root: pathlib.Path) -> None:
        (root / "config.json").write_text(
            json.dumps({"model_type": "openvla", "architectures": ["OpenVLAForActionPrediction"]})
        )
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.layer": "model-00001-of-00001.safetensors"}})
        )
        with open(root / "model-00001-of-00001.safetensors", "wb") as f:
            f.truncate(1024 * 1024 + 1)
        (root / "action_head--150000_checkpoint.pt").write_bytes(b"weights")
        (root / "proprio_projector--150000_checkpoint.pt").write_bytes(b"weights")
        for name in (
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer.json",
            "modeling_prismatic.py",
            "configuration_prismatic.py",
            "processing_prismatic.py",
        ):
            (root / name).write_text("{}")
        stats = {}
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            stats[suite] = {
                "action": {"q01": [0.0] * 7, "q99": [1.0] * 7},
                "proprio": {"q01": [0.0] * 8, "q99": [1.0] * 8},
            }
        (root / "dataset_statistics.json").write_text(json.dumps(stats))

    def test_accepts_complete_checkpoint_and_detects_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.make_checkpoint(root)
            self.assertEqual(detect_model_family(root), "openvla_oft")
            self.assertTrue(check_openvla_oft_model(root).ok)

    def test_rejects_missing_weight_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.make_checkpoint(root)
            (root / "model-00001-of-00001.safetensors").unlink()
            result = check_openvla_oft_model(root)
            self.assertFalse(result.ok)
            self.assertTrue(any("missing weight shard" in error for error in result.errors))

    def test_rejects_missing_suite_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.make_checkpoint(root)
            stats = json.loads((root / "dataset_statistics.json").read_text())
            stats.pop("libero_goal")
            (root / "dataset_statistics.json").write_text(json.dumps(stats))
            result = check_openvla_oft_model(root)
            self.assertTrue(any("missing suite 'libero_goal'" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
