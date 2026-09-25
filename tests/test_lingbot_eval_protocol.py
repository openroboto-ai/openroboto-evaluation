import pathlib
import sys
import unittest


_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "libero_eval"))

from lingbot_eval_protocol import derive_policy_seed, validate_policy_seed  # noqa: E402


class TestLingbotEvalProtocol(unittest.TestCase):
    def test_seed_is_stable_and_request_specific(self):
        first = derive_policy_seed(7, "libero_goal", 3, 11, 19)
        self.assertEqual(first, derive_policy_seed(7, "libero_goal", 3, 11, 19))
        self.assertNotEqual(first, derive_policy_seed(7, "libero_goal", 3, 11, 20))
        self.assertNotEqual(first, derive_policy_seed(7, "libero_goal", 3, 12, 19))
        self.assertEqual(validate_policy_seed(first), first)

    def test_invalid_seed_coordinates_are_rejected(self):
        for value in (True, -1, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                derive_policy_seed(value, "libero_goal", 0, 0, 0)
        with self.assertRaises(ValueError):
            derive_policy_seed(7, "", 0, 0, 0)
        for value in (True, -1, 1 << 63, "7"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_policy_seed(value)


if __name__ == "__main__":
    unittest.main()
