import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "benchmarks" / "axis_v1.0.json"
sys.path.insert(0, str(ROOT / "tools"))
from verify_axis_release import verify_release


class TestAxisV1Manifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_selected_tasks_are_a_basic_reproducible_subset(self):
        tasks = self.manifest["tasks"]
        task_ids = [task["task_id"] for task in tasks]
        self.assertEqual(self.manifest["name"], "axis_v1.0")
        self.assertEqual(self.manifest["status"], "runtime-ready")
        self.assertEqual(len(task_ids), 30)
        self.assertTrue({22, 57, 501, 757} <= set(task_ids))
        self.assertEqual(len(task_ids), len(set(task_ids)))
        verified = verify_release(MANIFEST_PATH, MANIFEST_PATH.with_name("axis_v1.0-tasks"))
        self.assertEqual(set(verified["tasks"]), {str(task_id) for task_id in task_ids})
        for task in tasks:
            self.assertTrue(task["instruction"])
            self.assertRegex(task["mjcf_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(task["checker_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(task["initial_state_sha256"], r"^[0-9a-f]{64}$")

    def test_runtime_contract_is_native_axis_joint_position_control(self):
        runtime = self.manifest["runtime"]
        self.assertEqual(runtime["mujoco_version"], "3.11.0")
        self.assertEqual(runtime["control_period_s"], 0.2)
        self.assertEqual(runtime["scene_policy"], "base-only")
        self.assertEqual(len(runtime["observation_joint_order"]), 9)
        self.assertEqual(
            runtime["observation_joint_order"][:2],
            [
                "franka/panda_finger_joint1",
                "franka/panda_finger_joint2",
            ],
        )
        self.assertFalse(self.manifest["protocol"]["randomization"])
        self.assertEqual(self.manifest["protocol"]["default_trials_per_task"], 20)


if __name__ == "__main__":
    unittest.main()
