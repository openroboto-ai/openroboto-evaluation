"""Replica-parallel Orbax writes must not be mistaken for truncated model tensors."""

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "libero_eval"))
import check_model


class ReplicaParallelCheckpointTests(unittest.TestCase):
    def test_stale_orbax_blobs_cannot_bypass_submission_size_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            blobs = root / "params" / "ocdbt.process_0" / "d"
            blobs.mkdir(parents=True)
            # Re-uploading into the same repo may retain past OCDBT blobs while
            # the live manifest still restores an ordinary FP32 Pi0.5 model.
            for name in ("current", "old"):
                with (blobs / name).open("wb") as file:
                    file.truncate(12_440_000_000)
            result = check_model.check_model(root, "pi05_axis_joint")
            self.assertFalse(result.ok)
            self.assertIn("model size limit exceeded", "\n".join(result.errors))

    def check_store(self, mesh_shape, byte_size):
        with tempfile.TemporaryDirectory() as temporary:
            params = pathlib.Path(temporary)
            spec = check_model.CONFIG_SPECS["pi05_axis_joint"]
            shapes = {("params", "PaliGemma", "weight"): [1_000_000_000]}
            shapes.update({
                ("params", *key): shape for key, shape in check_model._expected_proj_shapes(spec, torch=False).items()
            })
            (params / "_METADATA").write_text(
                json.dumps({
                    "tree_metadata": {
                        str(key): {"value_metadata": {"write_shape": shape}} for key, shape in shapes.items()
                    }
                })
            )
            (params / "_sharding").write_text(
                json.dumps({
                    "array-name": json.dumps({
                        "sharding_type": "NamedSharding",
                        "shape": mesh_shape,
                        "axis_names": ["batch", "fsdp"],
                        "partition_spec": [],
                    })
                })
            )
            (params / "manifest.ocdbt").write_bytes(b"manifest")
            (params / "d").mkdir()
            (params / "d" / "blob").write_bytes(b"data")
            result = check_model.CheckResult(str(params.parent), "pi05_axis_joint")
            with mock.patch.object(check_model, "_dir_total_bytes", return_value=byte_size):
                check_model._check_jax_params(params, spec, result)
            return result

    def test_three_replicas_defer_global_shape_count_to_restore(self):
        result = self.check_store([3, 1], 12_000_000_000)
        self.assertEqual(result.errors, [])
        self.assertTrue(any("deferred to Orbax restore" in warning for warning in result.warnings))

    def test_multi_device_metadata_does_not_allow_truncated_store(self):
        result = self.check_store([3, 1], 1_000_000)
        self.assertTrue(any("array data may be truncated" in error for error in result.errors))

    def test_single_device_still_checks_global_parameter_count(self):
        result = self.check_store([1, 1], 12_000_000_000)
        self.assertTrue(any("total parameter count" in error for error in result.errors))

    def test_invalid_mesh_shape_does_not_disable_parameter_count(self):
        result = self.check_store([0, 3], 12_000_000_000)
        self.assertTrue(any("total parameter count" in error for error in result.errors))
