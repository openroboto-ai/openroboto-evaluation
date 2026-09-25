import pathlib
import sys
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "libero_eval"))

from backbones import parse_backbone, resolve_backbone  # noqa: E402


class TestBackboneRegistry(unittest.TestCase):
    def test_aliases_normalize_to_stable_cli_names(self):
        self.assertEqual(parse_backbone("pi05").name, "pi0.5")
        self.assertEqual(parse_backbone("LingBot-VLA-2.0").name, "lingbot-vla-v2")
        self.assertEqual(parse_backbone("openvla_oft").name, "openvla-oft")

    def test_historical_default_is_pi05(self):
        selected, architectures = resolve_backbone(None)
        self.assertEqual(selected.name, "pi0.5")
        self.assertEqual(architectures, ("pi0.5",))

    def test_legacy_multi_architecture_allow_list_remains_supported(self):
        selected, architectures = resolve_backbone(None, ("pi0", "pi0.5"))
        self.assertEqual(selected.model_family, "openpi")
        self.assertEqual(architectures, ("pi0", "pi0.5"))

    def test_conflicting_new_and_legacy_flags_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_backbone("lingbot-vla-v2", ("pi0.5",))


if __name__ == "__main__":
    unittest.main()
