"""Exercise the pinned upstream optimizer with a tiny CPU-only model."""

import dataclasses
import importlib.util
import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "libero_eval"))


@unittest.skipUnless(importlib.util.find_spec("openpi"), "Requires the pinned OpenPI runtime")
class ScopeRuntimeTests(unittest.TestCase):
    def test_default_config_stays_full_and_dtype_conversion_cannot_be_enabled_for_expert_scope(self):
        from flax import nnx
        from openpi.training import config as upstream
        from axis_openpi_config import ActionExpertTrainConfig, make_config

        self.assertIs(type(make_config()), upstream.TrainConfig)
        config = make_config(training_scope="action-expert")
        self.assertIsInstance(config, ActionExpertTrainConfig)
        self.assertIsInstance(config.freeze_filter, nnx.Nothing)
        with self.assertRaisesRegex(ValueError, "empty dtype-conversion"):
            dataclasses.replace(config, freeze_filter=nnx.Everything())
        with self.assertRaisesRegex(ValueError, "training_scope"):
            make_config(training_scope="unknown")

    def test_upstream_initialization_and_update_preserve_frozen_float32_weights_exactly(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from flax import nnx, traverse_util
        from openpi.models import model as model_module
        from openpi.training import optimizer, weight_loaders
        from scripts import train
        from axis_openpi_config import ActionExpertTrainConfig
        from axis_training_scope import action_expert_parameter

        self.assertEqual({device.platform for device in jax.devices()}, {"cpu"})

        class Model(model_module.BaseModel):
            def __init__(self, rng):
                del rng
                self.action_dim, self.action_horizon, self.max_token_len = 2, 1, 1

                def weight():
                    return nnx.Dict(kernel=nnx.Param(jnp.array([1.0001234, -0.456789], dtype=jnp.float32)))

                self.PaliGemma = nnx.Dict(
                    img=weight(),
                    llm=nnx.Dict(embedder=weight(), layers=nnx.Dict(mlp=weight(), mlp_1=weight())),
                )
                self.action_in_proj, self.action_out_proj = weight(), weight()

            def compute_loss(self, rng, observation, actions, *, train=False):
                del rng, observation, actions, train
                return jnp.square(sum(jnp.sum(value) for value in jax.tree.leaves(nnx.state(self, nnx.Param))))[None]

            def sample_actions(self, rng, observation, **kwargs):
                del rng, kwargs
                return jnp.zeros((observation.state.shape[0], 1, 2))

        config = ActionExpertTrainConfig(
            name="tiny-axis-expert-scope",
            exp_name="cpu-regression-only",
            model=types.SimpleNamespace(create=Model),
            optimizer=optimizer.AdamW(),
            lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1, peak_lr=1e-3, decay_steps=10, decay_lr=1e-4),
            weight_loader=weight_loaders.NoOpWeightLoader(),
            ema_decay=None,
        )
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("batch", "fsdp"))
        state, _ = train.init_train_state(config, jax.random.key(73), mesh, resume=False)
        before = traverse_util.flatten_dict(state.params.to_pure_dict())
        expected = traverse_util.flatten_dict(nnx.state(Model(jax.random.key(0))).to_pure_dict())
        for path, value in before.items():
            self.assertEqual(value.dtype, jnp.float32)
            np.testing.assert_array_equal(np.asarray(value), np.asarray(expected[path]))
        observation = model_module.Observation(images={}, image_masks={}, state=jnp.zeros((1, 2)))
        after, _ = train.train_step(config, jax.random.key(73), state, (observation, jnp.zeros((1, 1, 2))))
        # The first warmup step has zero learning rate; the second must update.
        after, _ = train.train_step(config, jax.random.key(73), after, (observation, jnp.zeros((1, 1, 2))))
        after_values = traverse_util.flatten_dict(after.params.to_pure_dict())
        changed, frozen = 0, 0
        for path, value in before.items():
            if action_expert_parameter(path, None):
                self.assertFalse(np.array_equal(np.asarray(value), np.asarray(after_values[path])))
                changed += 1
            else:
                np.testing.assert_array_equal(np.asarray(value), np.asarray(after_values[path]))
                self.assertEqual(value.dtype, after_values[path].dtype)
                frozen += 1
        self.assertEqual((changed, frozen), (3, 3))


if __name__ == "__main__":
    unittest.main()
