import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_jax_runtime import axis_jax_runtime_metadata, configure_axis_jax_environment
from serve_axis_openpi import DeterministicAxisPolicy, main


class _ReachedOpenPI(RuntimeError):
    pass


class TestAxisJaxRuntime(unittest.TestCase):
    def test_cache_settings_cannot_leak_in_from_parent_environment(self):
        environment = {
            "JAX_COMPILATION_CACHE_DIR": "/old-cache",
            "JAX_ENABLE_COMPILATION_CACHE": "true",
            "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all",
            "XLA_FLAGS": "--xla_gpu_per_fusion_autotune_cache_dir=/old-tuning --xla_gpu_autotune_level=4",
            "CUDA_VISIBLE_DEVICES": "3",
        }
        configure_axis_jax_environment(environment)
        self.assertNotIn("JAX_COMPILATION_CACHE_DIR", environment)
        self.assertEqual(environment["JAX_ENABLE_COMPILATION_CACHE"], "false")
        self.assertEqual(environment["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"], "none")
        self.assertNotIn("/old-tuning", environment["XLA_FLAGS"])
        self.assertIn("--xla_gpu_autotune_level=0", environment["XLA_FLAGS"])
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "3")
        configured = dict(environment)
        configure_axis_jax_environment(environment)
        self.assertEqual(environment, configured)

    def test_direct_server_configures_jax_before_loading_openpi(self):
        def before_openpi(_root):
            self.assertEqual(os.environ["JAX_ENABLE_COMPILATION_CACHE"], "false")
            self.assertEqual(os.environ["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"], "none")
            self.assertNotIn("JAX_COMPILATION_CACHE_DIR", os.environ)
            self.assertIn("--xla_gpu_autotune_level=0", os.environ["XLA_FLAGS"])
            raise _ReachedOpenPI()

        with tempfile.TemporaryDirectory() as checkpoint, mock.patch.dict(
            os.environ,
            {
                "JAX_COMPILATION_CACHE_DIR": "/old-cache",
                "JAX_ENABLE_COMPILATION_CACHE": "true",
                "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all",
            },
        ), mock.patch.object(sys, "argv", ["serve_axis_openpi.py", "--checkpoint", checkpoint]), mock.patch(
            "axis_openpi_sources.bind_openpi_sources", side_effect=before_openpi
        ), self.assertRaises(_ReachedOpenPI):
            main()

    def test_pytorch_checkpoint_does_not_get_a_jax_runtime_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = pathlib.Path(temporary)
            (checkpoint / "model.safetensors").touch()
            with mock.patch.object(sys, "argv", ["serve_axis_openpi.py", "--checkpoint", str(checkpoint)]), mock.patch(
                "axis_openpi_sources.bind_openpi_sources", side_effect=_ReachedOpenPI
            ), mock.patch("axis_jax_runtime.configure_axis_jax_environment") as configure, self.assertRaises(
                _ReachedOpenPI
            ):
                main()
            configure.assert_not_called()

    def test_handshake_reports_the_numerical_runtime_without_claiming_hardware_parity(self):
        metadata = axis_jax_runtime_metadata()
        policy = DeterministicAxisPolicy(
            mock.Mock(metadata={}), seed=7, action_horizon=10, action_dim=32, numerical_runtime=metadata
        )
        self.assertEqual(policy.metadata["axis_numerical_runtime"], metadata)
        unconfigured = DeterministicAxisPolicy(mock.Mock(metadata={}), seed=7, action_horizon=10, action_dim=32)
        self.assertIsNone(unconfigured.metadata["axis_numerical_runtime"])
