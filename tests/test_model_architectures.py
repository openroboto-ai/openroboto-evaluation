"""Architecture allow-list parsing and automatic openpi config selection."""

import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "libero_eval"))

import check_model  # noqa: E402


def _result(config: str, *errors: str) -> check_model.CheckResult:
    return check_model.CheckResult(checkpoint_dir="/checkpoint", config=config, errors=list(errors))


class TestParseModelArchitectures(unittest.TestCase):
    def test_accepts_ascii_and_greek_spellings(self):
        self.assertEqual(check_model.parse_model_architectures("pi0,pi0.5"), ("pi0", "pi0.5"))
        self.assertEqual(check_model.parse_model_architectures("π0,π0.5"), ("pi0", "pi0.5"))

    def test_deduplicates_aliases_without_reordering(self):
        self.assertEqual(check_model.parse_model_architectures("pi05,pi0.5,pi0"), ("pi0.5", "pi0"))

    def test_rejects_special_both_value_and_empty_items(self):
        with self.assertRaisesRegex(ValueError, "unknown model architecture 'both'"):
            check_model.parse_model_architectures("both")
        with self.assertRaisesRegex(ValueError, "non-empty comma-separated"):
            check_model.parse_model_architectures("pi0,")


class TestModelConfigSelection(unittest.TestCase):
    def test_auto_selects_pi0_when_both_are_accepted(self):
        def fake_check(_path, config):
            if config == "pi0_libero":
                return _result(config)
            return _result(config, "wrong architecture")

        with mock.patch.object(check_model, "check_model", side_effect=fake_check):
            selected = check_model.check_model_for_architectures(
                "/checkpoint", ("pi0.5", "pi0")
            )

        self.assertTrue(selected.result.ok)
        self.assertEqual(selected.architecture, "pi0")
        self.assertEqual(selected.config, "pi0_libero")

    def test_valid_disallowed_pi0_is_rejected_by_filter(self):
        def fake_check(_path, config):
            if config == "pi0_libero":
                return _result(config)
            return _result(config, "wrong architecture")

        with mock.patch.object(check_model, "check_model", side_effect=fake_check):
            selected = check_model.check_model_for_architectures("/checkpoint", ("pi0.5",))

        self.assertFalse(selected.result.ok)
        self.assertEqual(selected.architecture, "pi0")
        self.assertIsNone(selected.config)
        self.assertEqual(
            selected.result.errors,
            ["checkpoint architecture 'pi0' is not accepted (accepted: pi0.5)"],
        )

    def test_explicit_config_must_be_in_allow_list(self):
        with mock.patch.object(check_model, "check_model") as validate:
            selected = check_model.check_model_for_architectures(
                "/checkpoint", ("pi0.5",), config="pi0_libero"
            )

        validate.assert_not_called()
        self.assertFalse(selected.result.ok)
        self.assertIn("is not accepted", selected.result.errors[0])


if __name__ == "__main__":
    unittest.main()
