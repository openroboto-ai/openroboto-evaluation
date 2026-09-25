"""EMA remains opt-in and cannot perturb frozen checkpoint parameters."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "libero_eval"))
from axis_training_ema import inference_parameters, validate_ema_decay


class EmaValidationTests(unittest.TestCase):
    def test_disabled_and_finite_decay_are_accepted(self):
        for value in (None, 0.5, 0.99, 0.999):
            validate_ema_decay(value)

    def test_invalid_decay_is_rejected(self):
        for value in (True, False, "0.9", float("nan"), float("inf"), -0.5, 0, 1, 1.01):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "ema_decay"):
                validate_ema_decay(value)

    def test_disabled_ema_returns_the_same_live_state_without_runtime_imports(self):
        params = object()
        self.assertIs(inference_parameters(params, None, None), params)


if __name__ == "__main__":
    unittest.main()
