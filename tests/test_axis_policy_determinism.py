import pathlib
import sys
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))

from serve_axis_openpi import DeterministicAxisPolicy  # noqa: E402


class _Policy:
    metadata = {"benchmark": "axis_v1.0"}

    def __init__(self) -> None:
        self.calls = []

    def infer(self, observation, *, noise):
        self.calls.append((observation, noise))
        return {"actions": noise[:, :9]}


class TestAxisPolicyDeterminism(unittest.TestCase):
    @staticmethod
    def _observation(call: int) -> dict:
        return {
            "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
            "observation/state": np.zeros(9),
            "prompt": "test",
            "_axis_policy_task_id": 501,
            "_axis_policy_trial": 2,
            "_axis_policy_call": call,
        }

    def test_noise_is_addressed_by_seed_task_trial_and_call(self):
        first_base = _Policy()
        second_base = _Policy()
        first = DeterministicAxisPolicy(first_base, seed=7, action_horizon=10, action_dim=32)
        second = DeterministicAxisPolicy(second_base, seed=7, action_horizon=10, action_dim=32)

        first.infer(self._observation(3))
        second.infer(self._observation(3))
        self.assertTrue(np.array_equal(first_base.calls[0][1], second_base.calls[0][1]))
        self.assertFalse(any(key.startswith("_axis_policy_") for key in first_base.calls[0][0]))

        first.infer(self._observation(4))
        self.assertFalse(np.array_equal(first_base.calls[0][1], first_base.calls[1][1]))
        self.assertEqual(first.metadata["axis_policy_seed"], 7)

    def test_missing_coordinates_fail_closed(self):
        policy = DeterministicAxisPolicy(_Policy(), seed=0, action_horizon=10, action_dim=32)
        with self.assertRaisesRegex(ValueError, "missing deterministic"):
            policy.infer({"prompt": "test"})

    def test_sample_ensemble_is_order_independent_and_trials_remain_distinct(self):
        base = _Policy()
        policy = DeterministicAxisPolicy(base, seed=7, action_horizon=10, action_dim=32, policy_samples=5)
        observation = self._observation(3)
        first = policy.infer(observation)["actions"]
        expected = np.mean(np.stack([noise[:, :9] for _, noise in base.calls]), axis=0, dtype=np.float64)
        np.testing.assert_array_equal(first, expected)
        self.assertEqual(len(base.calls), 5)
        self.assertFalse(np.array_equal(base.calls[0][1], base.calls[1][1]))
        unrelated = {**observation, "_axis_policy_task_id": 999}
        policy.infer(unrelated)
        np.testing.assert_array_equal(policy.infer(observation)["actions"], first)
        next_trial = {**observation, "_axis_policy_trial": 3}
        self.assertFalse(np.array_equal(policy.infer(next_trial)["actions"], first))
        self.assertEqual(policy.metadata["axis_policy_samples"], 5)
        single = DeterministicAxisPolicy(_Policy(), seed=7, action_horizon=10, action_dim=32)
        np.testing.assert_array_equal(single.infer(observation)["actions"], base.calls[0][1][:, :9])

    def test_invalid_sampling_budget_fails_before_inference(self):
        for count in (0, 17, 1.5, True):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "policy_samples"):
                DeterministicAxisPolicy(_Policy(), seed=7, action_horizon=10, action_dim=32, policy_samples=count)

    def test_medoid_returns_one_coherent_chunk_with_minimal_pairwise_squared_distance(self):
        base = _Policy()
        policy = DeterministicAxisPolicy(
            base,
            seed=7,
            action_horizon=10,
            action_dim=32,
            policy_samples=5,
            sample_reduction="medoid",
        )
        result = policy.infer(self._observation(3))["actions"]
        chunks = [noise[:, :9].astype(np.float64) for _, noise in base.calls]
        pairwise_costs = [sum(float(np.square(a - b).sum()) for b in chunks) for a in chunks]
        np.testing.assert_array_equal(result, chunks[int(np.argmin(pairwise_costs))])
        self.assertFalse(np.array_equal(result, np.mean(chunks, axis=0)))
        self.assertEqual(policy.metadata["axis_sample_reduction"], "medoid")


if __name__ == "__main__":
    unittest.main()
