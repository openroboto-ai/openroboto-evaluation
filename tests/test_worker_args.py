"""benchmark worker CLI 参数解析的回归测试。"""

import contextlib
import io
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmark_worker import worker  # noqa: E402


class TestQueueBaseModel(unittest.TestCase):
    def test_accepts_the_two_wire_values(self):
        for value in ("pi0.5", "lingbot-vla-2.0"):
            with self.subTest(value=value):
                self.assertEqual(worker.select_base_model({"base_model": value}), value)

    def test_rejects_missing_or_unsupported_values(self):
        for value in (None, "", "auto", "lingbot-vla-v2", "openvla-oft", " pi0.5 "):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "invalid base_model"):
                worker.select_base_model({"base_model": value})

    def test_robotwin_requires_lingbot(self):
        with self.assertRaisesRegex(ValueError, "robotwin tasks require"):
            worker.select_base_model({"base_model": "pi0.5"}, "robotwin")

    def test_axis_requires_pi05(self):
        self.assertEqual(worker.select_base_model({"base_model": "pi0.5"}, "axis_v1.0"), "pi0.5")
        with self.assertRaisesRegex(ValueError, "axis_v1.0 tasks require"):
            worker.select_base_model({"base_model": "lingbot-vla-2.0"}, "axis_v1.0")


class TestParseArgs(unittest.TestCase):
    @staticmethod
    def _base_argv() -> list[str]:
        return [
            "benchmark-worker",
            "--backend-url",
            "http://localhost:8001",
            "--public-api-key",
            "public",
            "--admin-api-key",
            "admin",
            "--download-strategies",
            "hub",
            "--benchmark",
            "libero",
            "--num-trials",
            "1",
        ]

    def test_worker_has_no_base_model_cli_override(self):
        for option, value in (("--backbone", "pi0.5"), ("--model-architectures", "pi0.5")):
            argv = [*self._base_argv(), option, value]
            with self.subTest(option=option), mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(SystemExit, "2"):
                    worker.parse_args()

    def test_base_model_is_not_stored_in_worker_args(self):
        argv = self._base_argv()
        with mock.patch.object(sys, "argv", argv):
            args = worker.parse_args()
        self.assertFalse(hasattr(args, "backbone"))
        self.assertEqual(args.max_batch, 4)

    def test_lingbot_server_impl_can_override_pi_server_impl(self):
        argv = [*self._base_argv(), "--server-impl", "batched", "--lingbot-server-impl", "static"]
        with mock.patch.object(sys, "argv", argv):
            args = worker.parse_args()
        self.assertEqual(args.server_impl, "batched")
        self.assertEqual(args.lingbot_server_impl, "static")

    def test_max_batch_must_be_a_positive_power_of_two(self):
        for value in ("0", "3", "-2"):
            with self.subTest(value=value), mock.patch.object(sys, "argv", [*self._base_argv(), "--max-batch", value]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(SystemExit, "2"):
                    worker.parse_args()

    def test_help_renders_literal_percent_sign(self):
        output = io.StringIO()
        with mock.patch.object(sys, "argv", ["benchmark-worker", "--help"]):
            with contextlib.redirect_stdout(output), self.assertRaisesRegex(SystemExit, "0"):
                worker.parse_args()

        self.assertIn("+143%", output.getvalue())

    def test_rejects_invalid_gpu_wait_settings(self):
        for flag, value in (("--gpu-max-used-mib", "-1"), ("--gpu-wait-interval", "0")):
            with self.subTest(flag=flag):
                with mock.patch.object(sys, "argv", ["benchmark-worker", flag, value]):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(SystemExit, "2"):
                        worker.parse_args()

    def test_robotwin_defaults_and_official_protocol_constraints(self):
        argv = [
            "benchmark-worker",
            "--backend-url",
            "http://localhost:8001",
            "--public-api-key",
            "public",
            "--admin-api-key",
            "admin",
            "--download-strategies",
            "hub",
            "--benchmark",
            "robotwin",
            "--num-trials",
            "100",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = worker.parse_args()
        self.assertEqual(args.workers_per_gpu, 1)
        self.assertTrue(args.no_init_randomization)

        invalid_cases = (
            ("wrong trial count", [*argv[:-1], "99"]),
            ("task subset", [*argv, "--task-ids", "0"]),
        )
        for label, invalid_argv in invalid_cases:
            with self.subTest(label=label), mock.patch.object(sys, "argv", invalid_argv):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(SystemExit, "2"):
                    worker.parse_args()

    def test_axis_defaults_and_frozen_protocol_constraints(self):
        argv = [
            "benchmark-worker",
            "--backend-url",
            "http://localhost:8001",
            "--public-api-key",
            "public",
            "--admin-api-key",
            "admin",
            "--download-strategies",
            "hub",
            "--benchmark",
            "axis_v1.0",
            "--num-trials",
            "20",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = worker.parse_args()
        self.assertEqual(args.workers_per_gpu, 1)
        self.assertEqual(args.eval_config, "pi05_axis_joint")
        self.assertTrue(args.no_init_randomization)

        invalid_cases = (
            ("wrong trial count", [*argv[:-1], "2"]),
            ("task subset", [*argv, "--task-ids", "501"]),
            ("batched server", [*argv, "--server-impl", "batched"]),
            ("wrong config", [*argv, "--eval-config", "pi05_libero"]),
        )
        for label, invalid_argv in invalid_cases:
            with self.subTest(label=label), mock.patch.object(sys, "argv", invalid_argv):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(SystemExit, "2"):
                    worker.parse_args()


class TestEvaluatorSourceCommit(unittest.TestCase):
    def test_local_reports_do_not_hide_dirty_runtime_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)

            def git(*arguments):
                return subprocess.run(
                    ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
                ).stdout.strip()

            git("init", "-q")
            (root / "benchmark_worker").mkdir()
            source = root / "benchmark_worker" / "worker.py"
            source.write_text("# committed runtime\n")
            (root / "docs").mkdir()
            (root / "docs" / "report.md").write_text("committed report\n")
            git("add", ".")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline")
            revision = git("rev-parse", "HEAD")
            (root / "docs" / "report.md").write_text("updated report\n")
            (root / "docs" / "new-report.csv").write_text("local,data\n")
            (root / "deploy" / "evidence" / "audit").mkdir(parents=True)
            (root / "deploy" / "evidence" / "audit" / "result.json").write_text("{}\n")
            with mock.patch.object(worker, "VALIDATOR_ROOT", root):
                self.assertEqual(worker.resolve_evaluator_source_commit(), revision)
                source.write_text("# changed runtime\n")
                with self.assertRaisesRegex(ValueError, "uncommitted source files"):
                    worker.resolve_evaluator_source_commit()
                git("checkout", "--", "benchmark_worker/worker.py")
                extra_source = root / "benchmark_worker" / "new_module.py"
                extra_source.write_text("# untracked runtime\n")
                with self.assertRaisesRegex(ValueError, "uncommitted source files"):
                    worker.resolve_evaluator_source_commit()
                extra_source.unlink()
                (root / "pyproject.toml").write_text("[project]\n")
                with self.assertRaisesRegex(ValueError, "uncommitted source files"):
                    worker.resolve_evaluator_source_commit()

    def test_explicit_immutable_bundle_revision(self):
        revision = "a" * 40
        self.assertEqual(worker.resolve_evaluator_source_commit(revision), revision)
        with self.assertRaisesRegex(ValueError, "full 40-character"):
            worker.resolve_evaluator_source_commit("main")

    def test_clean_checkout_uses_head(self):
        revision = "b" * 40
        completed = [
            subprocess.CompletedProcess(["git"], 0, stdout=revision + "\n"),
            subprocess.CompletedProcess(["git"], 0, stdout=""),
        ]
        with mock.patch("benchmark_worker.worker.subprocess.run", side_effect=completed):
            self.assertEqual(worker.resolve_evaluator_source_commit(), revision)

    def test_dirty_checkout_is_rejected(self):
        completed = [
            subprocess.CompletedProcess(["git"], 0, stdout="c" * 40 + "\n"),
            subprocess.CompletedProcess(["git"], 0, stdout=" M libero_eval/run_eval.py\n"),
        ]
        with mock.patch("benchmark_worker.worker.subprocess.run", side_effect=completed):
            with self.assertRaisesRegex(ValueError, "uncommitted source files"):
                worker.resolve_evaluator_source_commit()


if __name__ == "__main__":
    unittest.main()
