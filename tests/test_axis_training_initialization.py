"""Sharded initialization preserves weights, RNG, optimizer and EMA semantics."""

import dataclasses
import importlib.util
import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "libero_eval"))
from axis_training_initialization import init_train_state, select_parameter_sharding


class ParameterProjectionTests(unittest.TestCase):
    def test_partial_checkpoint_only_receives_matching_sharding_leaves(self):
        self.assertEqual(
            select_parameter_sharding({"layer": {"weight": object()}}, {"layer": {"weight": "shard", "bias": "rep"}}),
            {"layer": {"weight": "shard"}},
        )

    def test_unknown_checkpoint_key_fails(self):
        with self.assertRaises(KeyError):
            select_parameter_sharding({"missing": object()}, {})


@unittest.skipUnless(importlib.util.find_spec("openpi"), "Requires the pinned OpenPI runtime")
class RuntimeInitializationTests(unittest.TestCase):
    def test_four_device_initialization_matches_upstream(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from flax import nnx
        from openpi.training import config as training_config
        from openpi.training import optimizer
        from scripts import train

        if len(jax.devices()) < 4:
            self.skipTest("Run with XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu")

        class Model(nnx.Module):
            def __init__(self, rng):
                self.weight = nnx.Param(jnp.zeros((1024, 2048), dtype=jnp.float32))
                self.bias = nnx.Param(jax.random.normal(rng, (2048,)))

        class Loader:
            def load(self, shape):
                return {**shape, "weight": np.full((1024, 2048), 0.25, dtype=np.float32)}

        config = training_config.TrainConfig(
            name="toy",
            exp_name="initialization-equivalence",
            optimizer=optimizer.AdamW(),
            lr_schedule=optimizer.CosineDecaySchedule(),
            model=types.SimpleNamespace(create=Model),
            freeze_filter=lambda path, value: False,
            ema_decay=0.99,
            weight_loader=Loader(),
        )
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(1, 4), ("batch", "fsdp"))
        original, _ = train.init_train_state(config, jax.random.key(42), mesh, resume=False)
        changed, partition = init_train_state(config, jax.random.key(42), mesh, resume=False)
        jax.block_until_ready(changed)
        for name in ("params", "opt_state", "ema_params"):
            expected = jax.tree.leaves(getattr(original, name))
            actual = jax.tree.leaves(getattr(changed, name))
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        self.assertFalse(changed.params.weight.value.sharding.is_fully_replicated)
        self.assertEqual(changed.params.weight.value.sharding, partition.params.weight.value)
        np.testing.assert_array_equal(np.asarray(changed.params.weight.value), 0.25)
        shape, _ = init_train_state(
            dataclasses.replace(config, weight_loader=None), jax.random.key(42), mesh, resume=True
        )
        self.assertEqual(shape.params.weight.value.shape, (1024, 2048))


if __name__ == "__main__":
    unittest.main()
