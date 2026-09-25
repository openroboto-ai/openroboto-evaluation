import pathlib
import sys
import tempfile
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_vla import (  # noqa: E402
    AxisInputs,
    AxisOutputs,
    AxisReplayDataset,
    SCHEMA_VERSION,
    TRAINING_PURPOSE,
    load_artifact,
    save_artifact,
)


class TestAxisVlaArtifact(unittest.TestCase):
    def _metadata(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "purpose": TRAINING_PURPOSE,
            "eligible_for_scoring": False,
            "instruction": "Put the Brush in the Basket",
            "training_examples": 5,
            "successful_replays": 2,
        }

    def test_round_trip_and_chunks_do_not_cross_episodes(self):
        images = np.arange(5 * 4 * 6 * 3, dtype=np.uint8).reshape(5, 4, 6, 3)
        states = np.arange(45, dtype=np.float32).reshape(5, 9)
        actions = states + 100
        ends = np.asarray([2, 5], dtype=np.int64)
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "artifact.npz"
            saved = save_artifact(
                path,
                metadata=self._metadata(),
                images=images,
                states=states,
                actions=actions,
                episode_ends=ends,
            )
            loaded = load_artifact(path)
            dataset = AxisReplayDataset(path, action_horizon=3)
        self.assertEqual(saved.metadata["artifact_sha256"], loaded.metadata["artifact_sha256"])
        np.testing.assert_array_equal(dataset[1]["actions"], np.stack([actions[1], actions[1], actions[1]]))
        np.testing.assert_array_equal(dataset[3]["actions"], np.stack([actions[3], actions[4], actions[4]]))

    def test_digest_tampering_is_rejected(self):
        images = np.zeros((5, 4, 6, 3), dtype=np.uint8)
        states = np.zeros((5, 9), dtype=np.float32)
        actions = np.zeros((5, 9), dtype=np.float32)
        ends = np.asarray([2, 5], dtype=np.int64)
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "artifact.npz"
            save_artifact(
                path,
                metadata=self._metadata(),
                images=images,
                states=states,
                actions=actions,
                episode_ends=ends,
            )
            with np.load(path, allow_pickle=False) as payload:
                values = {key: payload[key] for key in payload.files}
            values["states"] = values["states"].copy()
            values["states"][0, 0] = 1
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_artifact(path)

    def test_multitask_prompt_and_action_chunks_follow_the_same_episode(self):
        metadata = {**self._metadata(), "episode_instructions": ["put brush in basket", "put cup in basket"]}
        actions = np.arange(45, dtype=np.float32).reshape(5, 9)
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "multi.npz"
            save_artifact(
                path,
                metadata=metadata,
                images=np.zeros((5, 4, 6, 3), dtype=np.uint8),
                states=np.zeros((5, 9), dtype=np.float32),
                actions=actions,
                episode_ends=np.asarray([2, 5]),
            )
            dataset = AxisReplayDataset(path, action_horizon=3)
            self.assertEqual(dataset[1]["prompt"], "put brush in basket")
            self.assertEqual(dataset[2]["prompt"], "put cup in basket")
            self.assertEqual(dataset[-1]["prompt"], "put cup in basket")
            np.testing.assert_array_equal(dataset[1]["actions"], np.stack([actions[1]] * 3))

    def test_multitask_instruction_count_must_match_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            for instructions in (["one"], ["one", ""], "not-a-list"):
                with self.subTest(instructions=instructions), self.assertRaisesRegex(
                    ValueError, "episode_instructions"
                ):
                    save_artifact(
                        pathlib.Path(temporary) / "bad.npz",
                        metadata={**self._metadata(), "episode_instructions": instructions},
                        images=np.zeros((5, 4, 6, 3), dtype=np.uint8),
                        states=np.zeros((5, 9), dtype=np.float32),
                        actions=np.zeros((5, 9), dtype=np.float32),
                        episode_ends=np.asarray([2, 5]),
                    )


class TestAxisOpenPiTransforms(unittest.TestCase):
    def test_inputs_mask_missing_wrist_views_and_preserve_native_contract(self):
        sample = AxisInputs()({
            "observation/image": np.zeros((3, 8, 6), dtype=np.uint8),
            "observation/state": np.arange(9, dtype=np.float32),
            "actions": np.zeros((10, 9), dtype=np.float32),
            "prompt": b"task",
        })
        self.assertEqual(sample["image"]["base_0_rgb"].shape, (8, 6, 3))
        self.assertTrue(sample["image_mask"]["base_0_rgb"])
        self.assertFalse(sample["image_mask"]["left_wrist_0_rgb"])
        self.assertFalse(sample["image_mask"]["right_wrist_0_rgb"])
        self.assertEqual(sample["state"].shape, (9,))
        self.assertEqual(sample["actions"].shape, (10, 9))
        self.assertEqual(sample["prompt"], "task")

    def test_outputs_remove_only_openpi_padding(self):
        actions = np.arange(3 * 32).reshape(3, 32)
        result = AxisOutputs()({"actions": actions})
        np.testing.assert_array_equal(result["actions"], actions[:, :9])

    def test_binary_gripper_decodes_physical_targets_without_mutating_arm_or_input(self):
        actions = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
        actions[:, :2] = [[0.004, 0.028], [0.014, 0.030], [-0.01, 0.01], [0.05, 0.05]]
        before = actions.copy()
        result = AxisOutputs("symmetric-binary")({"actions": actions})["actions"]
        np.testing.assert_allclose(result[:, :2], [[0, 0], [0.04, 0.04], [0, 0], [0.04, 0.04]])
        np.testing.assert_array_equal(result[:, 2:], actions[:, 2:9])
        np.testing.assert_array_equal(actions, before)

    def test_invalid_decoder_or_nonfinite_actions_fail(self):
        with self.assertRaisesRegex(ValueError, "gripper mode"):
            AxisOutputs("task-specific")
        for value in (np.inf, np.nan):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "non-finite"):
                AxisOutputs()({"actions": np.full((10, 32), value)})


if __name__ == "__main__":
    unittest.main()
