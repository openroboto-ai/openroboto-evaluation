import copy
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "libero_eval")]
from audit_axis_task_overlap import audit, predicate_implication  # noqa: E402
from axis_runtime import RuntimeState, evaluate_checker  # noqa: E402


class TestAxisPredicateOverlap(unittest.TestCase):
    def checker(self, task_id):
        payload = json.loads((ROOT / f"configs/benchmarks/axis_v1.0-tasks/{task_id}.json").read_text())
        return payload["checker_config"]["checker"]

    def state(self, position):
        return RuntimeState(
            positions={"brush_3": position, "basket_1": [0.0, 0.0, 0.0]},
            orientations={},
            joints={"franka/panda_finger_joint1": 0.04},
        )

    def test_inside_basket_is_also_accepted_by_the_beside_predicate(self):
        source, target = self.checker(501), self.checker(505)
        self.assertIsNotNone(predicate_implication(source, target))
        self.assertIsNone(predicate_implication(target, source))
        inside = self.state([0.02, 0.01, 0.01])
        self.assertTrue(evaluate_checker(source, inside, inside)[0])
        self.assertTrue(evaluate_checker(target, inside, inside)[0])
        beside = self.state([0.1, 0.0, 0.0])
        self.assertFalse(evaluate_checker(source, beside, beside)[0])
        self.assertTrue(evaluate_checker(target, beside, beside)[0])

    def test_narrower_box_or_stricter_gripper_does_not_follow_from_the_source(self):
        source, target = self.checker(501), self.checker(505)
        target["checkers"][0]["xRange"] = [-0.05, 0.05]
        self.assertIsNone(predicate_implication(source, target))
        counterexample = self.state([0.055, 0.0, 0.01])
        self.assertTrue(evaluate_checker(source, counterexample, counterexample)[0])
        self.assertFalse(evaluate_checker(target, counterexample, counterexample)[0])
        target = self.checker(505)
        target["checkers"][1]["threshold"] = 0.05
        self.assertIsNone(predicate_implication(source, target))

    def test_different_objects_and_nonconjunctions_are_not_claimed_as_proofs(self):
        source, target = self.checker(501), self.checker(505)
        target["checkers"][0]["objName"] = "cup_1"
        self.assertIsNone(predicate_implication(source, target))
        alternative = copy.deepcopy(source)
        alternative["operator"] = "OR"
        self.assertIsNone(predicate_implication(alternative, self.checker(505)))

    def test_frozen_pool_reports_the_two_known_strict_implications(self):
        report = audit(ROOT / "configs/benchmarks/axis_v1.0.json")
        self.assertTrue(
            {(501, 505), (503, 506)}
            <= {(row["source_task_id"], row["target_task_id"]) for row in report["implications"]}
        )
