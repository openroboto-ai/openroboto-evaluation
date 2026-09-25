import copy
import hashlib
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_randomization import (  # noqa: E402
    ALGORITHM,
    build_trial_plan,
    load_randomization_plan,
    select_variant,
)
from axis_runtime import canonical_json_sha256  # noqa: E402
from axis_task import _resolve_trial_payloads, run as run_axis_task  # noqa: E402


class TestAxisRandomization(unittest.TestCase):
    def _fixture(self, root: pathlib.Path):
        task_spec = {
            "task_id": 501,
            "instruction": "Put the Brush in the Basket",
        }
        variants = []
        for variant_id, x in (("layout-a", 0.1), ("layout-b", 0.2), ("layout-c", 0.3)):
            payload = {
                "id": 501,
                "name": task_spec["instruction"],
                "status": "frozen",
                "embodiment": "franka",
                "mjcf_xml": f"<mujoco model='{variant_id}'/>",
                "checker_config": {"type": "Synthetic", "variant": variant_id},
                "initial_state": {"objects": {"brush": {"pos": [x, 0.0, 0.0]}}},
            }
            payload_path = root / "payloads" / f"{variant_id}.json"
            payload_path.parent.mkdir(exist_ok=True)
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
            variants.append({
                "variant_id": variant_id,
                "payload_path": f"payloads/{variant_id}.json",
                "payload_canonical_sha256": canonical_json_sha256(payload),
                "mjcf_sha256": hashlib.sha256(payload["mjcf_xml"].encode()).hexdigest(),
                "checker_sha256": canonical_json_sha256(payload["checker_config"]),
                "initial_state_sha256": canonical_json_sha256(payload["initial_state"]),
                "dimensions": {"object_pose": {"brush_x": x}, "visual_scene": variant_id},
            })
        manifest = {
            "schema_version": 1,
            "benchmark": "axis_v99.0",
            "protocol_revision": "axis_v99.0_test",
            "seed_contract": {
                "source": "queue",
                "algorithm": ALGORITHM,
                "namespace": "axis_v99.0-test",
            },
            "tasks": [{"task_id": 501, "instruction": task_spec["instruction"], "variants": variants}],
        }
        manifest_path = root / "randomization.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, {501: task_spec}, manifest

    def _load(self, manifest_path, specs):
        return load_randomization_plan(
            manifest_path,
            expected_benchmark="axis_v99.0",
            expected_protocol_revision="axis_v99.0_test",
            benchmark_task_specs=specs,
        )

    def test_selection_is_deterministic_order_independent_and_auditable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            manifest_path, specs, manifest = self._fixture(root)
            plan = self._load(manifest_path, specs)
            first = [select_variant(plan, task_id=501, trial=trial, seed=42) for trial in range(20)]

            shuffled = copy.deepcopy(manifest)
            shuffled["tasks"][0]["variants"].reverse()
            manifest_path.write_text(json.dumps(shuffled), encoding="utf-8")
            reordered = self._load(manifest_path, specs)
            second = [select_variant(reordered, task_id=501, trial=trial, seed=42) for trial in range(20)]

        self.assertEqual([item.spec.variant_id for item in first], [item.spec.variant_id for item in second])
        self.assertEqual([item.selection_digest for item in first], [item.selection_digest for item in second])
        self.assertEqual({item.spec.variant_id for item in first}, {"layout-a", "layout-b", "layout-c"})
        provenance = first[0].provenance()
        self.assertEqual(provenance["algorithm"], ALGORITHM)
        self.assertEqual(provenance["seed"], 42)
        self.assertRegex(provenance["selection_digest"], r"^[0-9a-f]{64}$")

    def test_trial_plan_resolves_and_verifies_every_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            manifest_path, specs, _ = self._fixture(root)
            resolved = build_trial_plan(self._load(manifest_path, specs), task_id=501, num_trials=8, seed=9)
        self.assertEqual(len(resolved), 8)
        self.assertEqual([item.selection.trial for item in resolved], list(range(8)))
        self.assertTrue(all(item.payload["id"] == 501 for item in resolved))

    def test_axis_task_runtime_consumes_the_versioned_trial_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            randomization_path, specs, _ = self._fixture(root)
            manifest = {
                "name": "axis_v99.0",
                "protocol_revision": "axis_v99.0_test",
                "protocol": {"randomization": True},
                "runtime": {},
                "tasks": [specs[501]],
            }
            args = types.SimpleNamespace(
                manifest=str(root / "axis_v99.0.json"),
                num_trials=6,
                randomization_manifest=str(randomization_path),
                randomization_seed=17,
            )
            trials = _resolve_trial_payloads(args, manifest, specs[501], root / "cache")
        self.assertEqual(len(trials), 6)
        self.assertTrue(all(item.source == "frozen-randomization-snapshot" for item in trials))
        self.assertEqual([item.randomization["trial"] for item in trials], list(range(6)))
        self.assertTrue(all(item.randomization["seed"] == 17 for item in trials))

    def test_axis_rejects_randomization_arguments_before_runtime_setup(self):
        manifest = {
            "name": "axis_v1.0",
            "protocol": {"randomization": False},
            "runtime": {},
        }
        args = types.SimpleNamespace(
            randomization_manifest="variants.json",
            randomization_seed=1,
            num_trials=1,
        )
        with self.assertRaisesRegex(ValueError, "base-only"):
            _resolve_trial_payloads(args, manifest, {"task_id": 501}, pathlib.Path("unused"))

    def test_axis_task_prepares_and_smokes_every_selected_physical_variant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            randomization_path, specs, _ = self._fixture(root)
            benchmark_manifest = {
                "schema_version": 1,
                "name": "axis_v99.0",
                "status": "runtime-ready",
                "protocol_revision": "axis_v99.0_test",
                "runtime": {
                    "renderer_backend": "osmesa",
                    "asset_base_url": "https://assets.example.test",
                },
                "protocol": {"randomization": True, "max_control_steps_per_trial": 80},
                "tasks": [specs[501]],
            }
            benchmark_path = root / "axis_v99.0.json"
            benchmark_path.write_text(json.dumps(benchmark_manifest), encoding="utf-8")
            args = types.SimpleNamespace(
                manifest=str(benchmark_path),
                task_id=501,
                cache_root=str(root / "cache"),
                task_api_base_url=None,
                asset_base_url=None,
                asset_fetch_workers=1,
                refresh_task=False,
                randomization_manifest=str(randomization_path),
                randomization_seed=42,
                num_trials=20,
                prepare_only=False,
                dry_run=True,
                smoke_control_steps=1,
                smoke_render_frames=1,
            )
            asset_cache = mock.Mock()
            asset_cache.prepare_scene.side_effect = lambda task_id, _xml, scene_key: (
                root / f"{task_id}-{scene_key}.xml",
                {"xml_files": 1, "binary_files": 0},
            )
            environments = [mock.Mock() for _ in range(3)]
            with mock.patch.dict("os.environ", {"MUJOCO_GL": "osmesa"}), mock.patch(
                "axis_task.AssetCache", return_value=asset_cache
            ), mock.patch("axis_task.AxisEnvironment", side_effect=environments) as environment_class, mock.patch(
                "axis_task._smoke_environment", return_value={"mujoco": {}, "smoke": {}}
            ):
                result = run_axis_task(args)

        self.assertEqual(result["benchmark"], "axis_v99.0")
        self.assertTrue(result["randomization"])
        self.assertEqual(len(result["variant_smoke"]), 3)
        self.assertEqual(asset_cache.prepare_scene.call_count, 3)
        self.assertEqual(environment_class.call_count, 3)
        self.assertTrue(all(environment.close.called for environment in environments))

    def test_payload_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            manifest_path, specs, _ = self._fixture(root)
            plan = self._load(manifest_path, specs)
            selected = select_variant(plan, task_id=501, trial=0, seed=1)
            selected.spec.payload_path.write_text('{"id":501}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "name drifted"):
                build_trial_plan(plan, task_id=501, num_trials=1, seed=1)

    def test_manifest_rejects_missing_coverage_duplicate_variants_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            manifest_path, specs, manifest = self._fixture(root)

            missing = copy.deepcopy(manifest)
            missing["tasks"] = []
            manifest_path.write_text(json.dumps(missing), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                self._load(manifest_path, specs)

            duplicate = copy.deepcopy(manifest)
            duplicate["tasks"][0]["variants"][1]["variant_id"] = "layout-a"
            manifest_path.write_text(json.dumps(duplicate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "repeats variant_id"):
                self._load(manifest_path, specs)

            escaping = copy.deepcopy(manifest)
            escaping["tasks"][0]["variants"][0]["payload_path"] = "../outside.json"
            manifest_path.write_text(json.dumps(escaping), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stay inside"):
                self._load(manifest_path, specs)

    def test_manifest_rejects_unknown_fields_duplicate_json_keys_and_fake_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            manifest_path, specs, manifest = self._fixture(root)

            unknown = copy.deepcopy(manifest)
            unknown["implicit_magic"] = True
            manifest_path.write_text(json.dumps(unknown), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fields must be exactly"):
                self._load(manifest_path, specs)

            manifest_path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "repeats key"):
                self._load(manifest_path, specs)

            fake = copy.deepcopy(manifest)
            for variant in fake["tasks"][0]["variants"][1:]:
                for field in ("mjcf_sha256", "checker_sha256", "initial_state_sha256"):
                    variant[field] = fake["tasks"][0]["variants"][0][field]
            manifest_path.write_text(json.dumps(fake), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not differ"):
                self._load(manifest_path, specs)

    def test_seed_and_trial_boundaries_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            manifest_path, specs, _ = self._fixture(root)
            plan = self._load(manifest_path, specs)
            for seed in (-1, 2**64, True, 1.5):
                with self.assertRaisesRegex(ValueError, "unsigned 64-bit"):
                    select_variant(plan, task_id=501, trial=0, seed=seed)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                select_variant(plan, task_id=501, trial=-1, seed=0)


if __name__ == "__main__":
    unittest.main()
