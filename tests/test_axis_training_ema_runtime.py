"""Exercise averaged checkpoint export with the pinned OpenPI CPU runtime."""

import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "libero_eval"))


@unittest.skipUnless(importlib.util.find_spec("openpi"), "Requires the pinned OpenPI runtime")
class EmaRuntimeTests(unittest.TestCase):
    def test_config_preserves_disabled_default_and_accepts_explicit_decay(self):
        from axis_openpi_config import make_config

        self.assertIsNone(make_config().ema_decay)
        self.assertIsNone(make_config(training_scope="action-expert").ema_decay)
        for scope in ("full", "action-expert"):
            self.assertEqual(make_config(training_scope=scope, ema_decay=0.99).ema_decay, 0.99)
            with self.assertRaisesRegex(ValueError, "ema_decay"):
                make_config(training_scope=scope, ema_decay=float("nan"))

    def test_export_replaces_drifted_frozen_ema_and_roundtrips_full_checkpoint(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        import orbax.checkpoint as ocp
        from flax import nnx
        from axis_openpi_config import make_config
        from axis_training_ema import inference_parameters

        self.assertEqual({device.platform for device in jax.devices()}, {"cpu"})
        live = nnx.state(
            nnx.Dict(
                PaliGemma=nnx.Dict(img=nnx.Param(jnp.array([1.0001234, -0.456789], dtype=jnp.float32))),
                action_in_proj=nnx.Param(jnp.array([3.0, 4.0], dtype=jnp.float32)),
            )
        )
        averaged = nnx.state(
            nnx.Dict(
                PaliGemma=nnx.Dict(img=nnx.Param(jnp.array([9.0, 8.0], dtype=jnp.bfloat16))),
                action_in_proj=nnx.Param(jnp.array([1.5, 2.5], dtype=jnp.float32)),
            )
        )
        selected = inference_parameters(live, averaged, make_config(training_scope="action-expert").trainable_filter)
        np.testing.assert_array_equal(selected.PaliGemma.img.value, live.PaliGemma.img.value)
        self.assertEqual(selected.PaliGemma.img.value.dtype, jnp.float32)
        np.testing.assert_array_equal(selected.action_in_proj.value, averaged.action_in_proj.value)
        np.testing.assert_array_equal(live.action_in_proj.value, [3.0, 4.0])
        full = inference_parameters(live, averaged, make_config().trainable_filter)
        np.testing.assert_array_equal(full.PaliGemma.img.value, averaged.PaliGemma.img.value)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "params"
            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(path, {"params": selected})
            restored = checkpointer.restore(path)["params"]
            np.testing.assert_array_equal(restored["PaliGemma"]["img"]["value"], live.PaliGemma.img.value)
            np.testing.assert_array_equal(restored["action_in_proj"]["value"], averaged.action_in_proj.value)
            checkpointer.close()

    def test_real_upstream_updates_produce_expected_ema_without_changing_frozen_weights(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from flax import nnx
        from openpi.models import model as model_module
        from openpi.training import optimizer, weight_loaders
        from scripts import train
        from axis_openpi_config import ActionExpertTrainConfig
        from axis_training_ema import inference_parameters

        class Model(model_module.BaseModel):
            def __init__(self, rng):
                del rng
                self.action_dim, self.action_horizon, self.max_token_len = 2, 1, 1
                self.PaliGemma = nnx.Dict(img=nnx.Dict(kernel=nnx.Param(jnp.array([1.0001234, -0.456789]))))
                self.action_in_proj = nnx.Dict(kernel=nnx.Param(jnp.array([0.25, 0.5])))

            def compute_loss(self, rng, observation, actions, *, train=False):
                del rng, observation, actions, train
                return jnp.square(sum(jnp.sum(v) for v in jax.tree.leaves(nnx.state(self, nnx.Param))))[None]

            def sample_actions(self, rng, observation, **kwargs):
                del rng, kwargs
                return jnp.zeros((observation.state.shape[0], 1, 2))

        self.assertEqual({device.platform for device in jax.devices()}, {"cpu"})
        config = ActionExpertTrainConfig(
            name="tiny-axis-ema",
            exp_name="cpu-regression-only",
            model=types.SimpleNamespace(create=Model),
            optimizer=optimizer.AdamW(),
            lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1, peak_lr=1e-3, decay_steps=10, decay_lr=1e-4),
            weight_loader=weight_loaders.NoOpWeightLoader(),
            ema_decay=0.9,
        )
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("batch", "fsdp"))
        state, _ = train.init_train_state(config, jax.random.key(73), mesh, resume=False)
        frozen = np.asarray(state.params.PaliGemma.img.kernel.value).copy()
        expected = np.asarray(state.params.action_in_proj.kernel.value).copy()
        observation = model_module.Observation(images={}, image_masks={}, state=jnp.zeros((1, 2)))
        for _ in range(4):
            state, _ = train.train_step(config, jax.random.key(73), state, (observation, jnp.zeros((1, 1, 2))))
            expected = 0.9 * expected + 0.1 * np.asarray(state.params.action_in_proj.kernel.value)
        selected = inference_parameters(state.params, state.ema_params, config.trainable_filter)
        np.testing.assert_allclose(selected.action_in_proj.kernel.value, expected, rtol=1e-6, atol=1e-7)
        self.assertFalse(np.array_equal(selected.action_in_proj.kernel.value, state.params.action_in_proj.kernel.value))
        np.testing.assert_array_equal(selected.PaliGemma.img.kernel.value, frozen)


if __name__ == "__main__":
    unittest.main()
