import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
UNIT = ROOT / "deploy" / "systemd" / "openroboto-axis-rotation@.service"
ENV_EXAMPLE = ROOT / "deploy" / "systemd" / "axis-rotation.env.example"


class TestAxisServiceTemplate(unittest.TestCase):
    def test_secrets_are_only_loaded_from_environment_file(self):
        unit = UNIT.read_text(encoding="utf-8")
        self.assertIn("EnvironmentFile=%h/.config/openroboto/axis-rotation-%i.env", unit)
        self.assertNotIn("--public-api-key", unit)
        self.assertNotIn("--admin-api-key", unit)
        self.assertNotIn("replace-me", unit)
        self.assertIn("UMask=0077", unit)

    def test_rotation_preserves_queue_version_and_frozen_axis_protocol(self):
        unit = UNIT.read_text(encoding="utf-8")
        self.assertNotIn("--benchmark ", unit)
        self.assertIn("--num-trials 20", unit)
        self.assertIn("--workers-per-gpu 1", unit)
        self.assertIn("--server-impl upstream", unit)
        self.assertIn("EVALUATOR_SOURCE_GIT_COMMIT", ENV_EXAMPLE.read_text(encoding="utf-8"))
        self.assertIn("--axis-runtime-pool ${AXIS_RUNTIME_POOL}", unit)
        self.assertIn("--axis-selector-root ${AXIS_SELECTOR_ROOT}", unit)
        self.assertIn("--axis-benchmark-dir ${AXIS_BENCHMARK_DIR}", unit)


if __name__ == "__main__":
    unittest.main()
