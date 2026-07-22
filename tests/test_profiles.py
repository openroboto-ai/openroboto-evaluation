"""Benchmark runtime/profile separation tests."""

import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmark_worker.profiles import (  # noqa: E402
    PLUS_SUITE_TASK_COUNTS,
    PRO_TARGETS,
    get_profile,
    target_from_env_name,
)
from benchmark_worker import worker  # noqa: E402


class TestProfiles(unittest.TestCase):
    def test_standard_and_custom_run_the_same_sixteen_suites(self):
        standard = get_profile("libero_pro")
        custom = get_profile("libero_pro_custom_1")
        self.assertEqual(standard.runtime_benchmark, "libero_pro")
        self.assertEqual(custom.runtime_benchmark, "libero_pro")
        self.assertEqual(standard.targets, custom.targets)
        self.assertEqual(len(custom.targets), 16)
        self.assertEqual(sum(standard.weights), 16.0)
        self.assertEqual(sum(custom.weights), 16.0)

    def test_custom_runtime_does_not_duplicate_or_reweight_suites(self):
        custom = get_profile("libero_pro_custom_1")
        self.assertTrue(all(weight == 1.0 for weight in custom.weights))

    def test_libero_plus_uses_official_suite_task_counts(self):
        profile = get_profile("libero_plus")
        self.assertEqual(
            profile.weights,
            tuple(float(PLUS_SUITE_TASK_COUNTS[target.base_suite]) for target in profile.targets),
        )
        self.assertEqual(sum(profile.weights), 10030.0)

    def test_runtime_name_round_trip(self):
        for target in PRO_TARGETS:
            self.assertEqual(target_from_env_name(target.env_name), target)
        with self.assertRaises(ValueError):
            target_from_env_name("libero_object_unknown")

    def test_worker_translates_custom_profile_to_pro_runtime(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            out_dir = pathlib.Path(tmp_str)
            (out_dir / "summary.json").write_text('{"tasks":{},"suites":{}}')
            args = types.SimpleNamespace(
                benchmark="libero_pro_custom_1",
                eval_config=None,
                num_trials=50,
                gpus="0",
                workers_per_gpu=1,
                init_workers_per_gpu=1,
                server_impl="upstream",
                task_ids="",
                eval_timeout=60,
            )
            proc = mock.Mock(returncode=0)
            proc.poll.return_value = 0
            worker.stop_event.clear()
            with mock.patch.object(worker.subprocess, "Popen", return_value=proc) as popen:
                worker.run_evaluation(
                    {"hf_commit": "a" * 40},
                    out_dir / "model",
                    out_dir,
                    args,
                )
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--benchmark") + 1], "libero_pro")

    def test_worker_only_accepts_complete_official_libero_plus_summary(self):
        args = types.SimpleNamespace(
            benchmark="libero_plus",
            eval_config=None,
            num_trials=1,
            gpus="0",
            workers_per_gpu=1,
            init_workers_per_gpu=1,
            server_impl="upstream",
            task_ids="",
            eval_timeout=60,
        )
        proc = mock.Mock(returncode=0)
        proc.poll.return_value = 0
        worker.stop_event.clear()
        with tempfile.TemporaryDirectory() as tmp_str:
            out_dir = pathlib.Path(tmp_str)
            (out_dir / "summary.json").write_text(
                '{"tasks":{},"suites":{},"evaluation_protocol":{"official_result":true}}'
            )
            with mock.patch.object(worker.subprocess, "Popen", return_value=proc) as popen:
                worker.run_evaluation(
                    {"hf_commit": "a" * 40},
                    out_dir / "model",
                    out_dir,
                    args,
                    init_seed=123,
                )
            self.assertNotIn("--init-seed", popen.call_args.args[0])

            (out_dir / "summary.json").write_text(
                '{"tasks":{},"suites":{},"evaluation_protocol":'
                '{"official_result":false,"deviations":["development subset"]}}'
            )
            with mock.patch.object(worker.subprocess, "Popen", return_value=proc):
                with self.assertRaisesRegex(worker.EvalInfrastructureError, "development subset"):
                    worker.run_evaluation(
                        {"hf_commit": "a" * 40},
                        out_dir / "model",
                        out_dir,
                        args,
                    )

    def test_worker_cli_enforces_libero_plus_protocol(self):
        base_argv = [
            "worker.py",
            "--backend-url",
            "http://localhost:8001",
            "--benchmark",
            "libero_plus",
            "--num-trials",
            "1",
        ]
        with mock.patch.object(sys, "argv", base_argv):
            args = worker.parse_args()
        self.assertTrue(args.no_init_randomization)

        with mock.patch.object(sys, "argv", [*base_argv[:-1], "2"]):
            with self.assertRaises(SystemExit):
                worker.parse_args()
        with mock.patch.object(sys, "argv", [*base_argv, "--task-ids", "0,1"]):
            with self.assertRaises(SystemExit):
                worker.parse_args()


if __name__ == "__main__":
    unittest.main()
