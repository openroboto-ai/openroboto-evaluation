import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))
sys.path.insert(0, str(ROOT / "tools"))

from axis_runtime import (  # noqa: E402
    RuntimeState,
    evaluate_checker,
    load_manifest,
    normalize_asset_reference,
    resolve_task_payload,
    task_specs,
)
from verify_axis_release import verify_release  # noqa: E402


class TestAxisRuntime(unittest.TestCase):
    def test_axis_pins_osmesa_renderer(self):
        self.assertEqual(load_manifest()["runtime"]["renderer_backend"], "osmesa")

    def test_asset_reference_is_resolved_relative_to_scene(self):
        self.assertEqual(
            normalize_asset_reference("scenes/task.xml", "../robots/franka/hand.stl"),
            "robots/franka/hand.stl",
        )
        with self.assertRaisesRegex(ValueError, "escapes cache root"):
            normalize_asset_reference("scene.xml", "../../secret")

    def test_axis_containment_and_gripper_checker(self):
        current = RuntimeState(
            positions={"brush_3": [0.01, 0.01, 0.01], "basket_1": [0.0, 0.0, 0.0]},
            orientations={},
            joints={"franka/panda_finger_joint1": 0.04, "franka/panda_finger_joint2": 0.04},
        )
        config = {
            "type": "CompositeChecker",
            "operator": "AND",
            "checkers": [
                {
                    "type": "RelativeCylinderChecker",
                    "objName": "brush_3",
                    "refName": "basket_1",
                    "xyRadius": 0.06,
                    "heightMin": -0.02,
                    "heightMax": 0.04,
                },
                {"type": "GripperOpenChecker", "threshold": 0.04},
            ],
        }
        passed, detail = evaluate_checker(config, current, current)
        self.assertTrue(passed)
        self.assertTrue(detail["passed"])

    def test_axis_directed_rotation_uses_runtime_initial(self):
        initial = RuntimeState({}, {"phone_0": [0.0, 0.0, 0.0, 1.0]}, {})
        half_angle = -(90.0 / 2.0) * 3.141592653589793 / 180.0
        current = RuntimeState(
            {}, {"phone_0": [0.0, 0.0, __import__("math").sin(half_angle), __import__("math").cos(half_angle)]}, {}
        )
        passed, detail = evaluate_checker(
            {
                "type": "DirectedRotationChecker",
                "bodyName": "phone_0",
                "targetAngleDeg": -90,
                "angleToleranceDeg": 10,
                "maxTiltDeg": 15,
            },
            current,
            initial,
        )
        self.assertTrue(passed)
        self.assertAlmostEqual(detail["twist_angle_deg"], -90.0)

    def test_unknown_checker_fails_closed(self):
        empty = RuntimeState({}, {}, {})
        with self.assertRaisesRegex(ValueError, "does not implement"):
            evaluate_checker({"type": "FutureChecker"}, empty, empty)

    def test_frozen_snapshot_is_preferred_over_network_and_populates_cache(self):
        manifest = load_manifest()
        spec = task_specs(manifest)[501]
        snapshot = ROOT / "configs" / "benchmarks" / "axis_v1.0-tasks" / "501.json"
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "httpx.get", side_effect=AssertionError("network must not be used")
        ):
            resolved = resolve_task_payload(
                spec,
                api_base_url="https://example.invalid/api",
                selection_contract=2,
                cache_root=pathlib.Path(temporary),
            )
            cached = pathlib.Path(temporary) / "tasks" / "501.json"
            self.assertTrue(cached.is_file())
            self.assertEqual(json.loads(cached.read_text()), json.loads(snapshot.read_text()))
        self.assertEqual(resolved.source, "frozen-snapshot")
        self.assertRegex(resolved.canonical_sha256, r"^[0-9a-f]{64}$")

    def test_release_snapshot_set_is_complete_and_hash_verified(self):
        manifest = load_manifest()
        specs = task_specs(manifest)
        snapshot_root = ROOT / "configs" / "benchmarks" / "axis_v1.0-tasks"
        self.assertEqual(
            sorted(path.name for path in snapshot_root.glob("*.json")),
            sorted(f"{task_id}.json" for task_id in specs),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for task_id, spec in specs.items():
                resolved = resolve_task_payload(
                    spec,
                    api_base_url="https://example.invalid/api",
                    selection_contract=2,
                    cache_root=pathlib.Path(temporary),
                )
                self.assertEqual(resolved.data["id"], task_id)
                self.assertEqual(resolved.source, "frozen-snapshot")

        release = verify_release(ROOT / "configs" / "benchmarks" / "axis_v1.0.json", snapshot_root)
        self.assertEqual(release["benchmark"], "axis_v1.0")
        self.assertRegex(str(release["snapshot_bundle_canonical_sha256"]), r"^[0-9a-f]{64}$")

    def test_missing_snapshot_fails_closed_instead_of_falling_back_to_api(self):
        spec = task_specs(load_manifest())[501]
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "httpx.get", side_effect=AssertionError("network fallback must not occur")
        ):
            snapshot_root = pathlib.Path(temporary) / "snapshots"
            snapshot_root.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "frozen AXIS task snapshot is missing"):
                resolve_task_payload(
                    spec,
                    api_base_url="https://example.invalid/api",
                    selection_contract=2,
                    cache_root=pathlib.Path(temporary) / "cache",
                    snapshot_root=snapshot_root,
                )

    def test_refresh_is_an_explicit_upstream_drift_audit(self):
        spec = task_specs(load_manifest())[501]
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"id": 501, "name": spec["instruction"]}
        with tempfile.TemporaryDirectory() as temporary, mock.patch("httpx.get", return_value=response) as get:
            with self.assertRaisesRegex(ValueError, "runtime drifted"):
                resolve_task_payload(
                    spec,
                    api_base_url="https://api.example.test",
                    selection_contract=2,
                    cache_root=pathlib.Path(temporary),
                    refresh=True,
                )
        get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
