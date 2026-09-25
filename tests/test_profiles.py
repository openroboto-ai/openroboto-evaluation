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
from test_axis_yaml_benchmark import TASK_IDS, complete_summary


def _complete_summary(task_count: int, trials: int, official_result: bool | None = None) -> str:
    tasks = {
        f"suite_task{i:05d}": {
            "status": "ok",
            "task_suite_name": "suite",
            "task_id": i,
            "num_trials": trials,
        }
        for i in range(task_count)
    }
    summary = {"tasks": tasks, "suites": {}, "num_trials_per_task": trials}
    if official_result is not None:
        summary["evaluation_protocol"] = {"official_result": official_result, "deviations": []}
    return __import__("json").dumps(summary)


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

    def test_robotwin_profile_is_the_official_fifty_task_clean_protocol(self):
        profile = get_profile("robotwin")
        self.assertEqual(profile.runtime_benchmark, "robotwin")
        self.assertEqual(profile.expected_task_count, 50)
        self.assertEqual(tuple(target.env_name for target in profile.targets), ("robotwin_clean",))

    def test_axis_profile_pins_thirty_native_joint_tasks(self):
        profile = get_profile("axis_v1.0")
        self.assertEqual(profile.runtime_benchmark, "axis_v1.0")
        self.assertEqual(profile.expected_task_count, 30)
        self.assertEqual(profile.expected_task_ids, tuple(TASK_IDS))
        self.assertEqual(profile.policy_seed, 20260907)
        self.assertEqual(tuple(target.env_name for target in profile.targets), ("axis_v1.0",))

    def test_queue_benchmark_and_optional_revision(self):
        revision = "axis_v1.0_30tasks_native_joint_osmesa_v1"
        self.assertEqual(
            worker.select_benchmark({"benchmark": "axis_v1.0", "protocol_revision": revision}), "axis_v1.0"
        )
        self.assertEqual(worker.select_benchmark({"benchmark": "axis_v1.0"}), "axis_v1.0")
        with self.assertRaisesRegex(ValueError, "must include benchmark"):
            worker.select_benchmark({})
        with self.assertRaisesRegex(ValueError, "protocol_revision"):
            worker.select_benchmark({"benchmark": "axis_v1.0", "protocol_revision": "old"})
        self.assertEqual(worker.select_benchmark({}, "libero"), "libero")
        self.assertEqual(worker.select_benchmark({"protocol_revision": revision}, "libero"), "libero")

    def test_runtime_name_round_trip(self):
        for target in PRO_TARGETS:
            self.assertEqual(target_from_env_name(target.env_name), target)
        with self.assertRaises(ValueError):
            target_from_env_name("libero_object_unknown")

    def test_worker_translates_custom_profile_to_pro_runtime(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            out_dir = pathlib.Path(tmp_str)
            (out_dir / "summary.json").write_text(_complete_summary(160, 50))
            args = types.SimpleNamespace(
                benchmark="libero_pro_custom_1",
                evaluator_source_git_commit="d" * 40,
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
                    {"hf_commit": "a" * 40, "base_model": "pi0.5"},
                    out_dir / "model",
                    out_dir,
                    args,
                )
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--benchmark") + 1], "libero_pro")
            self.assertEqual(command[command.index("--evaluator-source-git-commit") + 1], "d" * 40)

    def test_worker_passes_queue_base_model_to_run_eval(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            out_dir = pathlib.Path(tmp_str)
            (out_dir / "summary.json").write_text(_complete_summary(40, 1))
            args = types.SimpleNamespace(
                benchmark="libero",
                lingbot_norm_stats=None,
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
            with mock.patch.object(worker.subprocess, "Popen", return_value=proc) as popen:
                worker.run_evaluation(
                    {"hf_commit": "a" * 40, "base_model": "lingbot-vla-2.0"},
                    out_dir / "model",
                    out_dir,
                    args,
                )
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--backbone") + 1], "lingbot-vla-2.0")
            self.assertNotIn("--model-architectures", command)

    def test_worker_runs_complete_axis_profile_with_native_config(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            out_dir = pathlib.Path(tmp_str)
            summary = complete_summary()
            (out_dir / "summary.json").write_text(__import__("json").dumps(summary))
            args = types.SimpleNamespace(
                benchmark="axis_v1.0",
                evaluator_source_git_commit="d" * 40,
                eval_config="pi05_axis_joint",
                num_trials=20,
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
                summary, error = worker.run_evaluation(
                    {"hf_commit": "a" * 40, "base_model": "pi0.5"},
                    out_dir / "model",
                    out_dir,
                    args,
                    init_seed=123,
                )
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--benchmark") + 1], "axis_v1.0")
            self.assertEqual(command[command.index("--config") + 1], "pi05_axis_joint")
            self.assertEqual(command[command.index("--seed") + 1], "20260907")
            self.assertEqual(command[command.index("--workers-per-gpu") + 1], "1")
            self.assertNotIn("--init-seed", command)
            self.assertEqual(len(summary["tasks"]), 30)
            self.assertEqual(error, "")

    def test_worker_applies_static_server_override_only_to_lingbot(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            out_dir = pathlib.Path(tmp_str)
            (out_dir / "summary.json").write_text(_complete_summary(40, 1))
            args = types.SimpleNamespace(
                benchmark="libero",
                lingbot_norm_stats=None,
                lingbot_server_impl="static",
                max_batch=4,
                eval_config=None,
                num_trials=1,
                gpus="0",
                workers_per_gpu=8,
                init_workers_per_gpu=1,
                server_impl="batched",
                task_ids="",
                eval_timeout=60,
            )
            proc = mock.Mock(returncode=0)
            proc.poll.return_value = 0
            worker.stop_event.clear()
            with mock.patch.object(worker.subprocess, "Popen", return_value=proc) as popen:
                worker.run_evaluation(
                    {"hf_commit": "a" * 40, "base_model": "lingbot-vla-2.0"},
                    out_dir / "model",
                    out_dir,
                    args,
                )
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--server-impl") + 1], "static")
            self.assertEqual(command[command.index("--max-batch") + 1], "4")

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
            (out_dir / "summary.json").write_text(_complete_summary(10030, 1, official_result=True))
            with mock.patch.object(worker.subprocess, "Popen", return_value=proc) as popen:
                worker.run_evaluation(
                    {"hf_commit": "a" * 40, "base_model": "pi0.5"},
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
                        {"hf_commit": "a" * 40, "base_model": "pi0.5"},
                        out_dir / "model",
                        out_dir,
                        args,
                    )

    def test_worker_cli_enforces_libero_plus_protocol(self):
        base_argv = [
            "worker.py",
            "--backend-url",
            "http://localhost:8001",
            "--public-api-key",
            "public",
            "--admin-api-key",
            "admin",
            "--benchmark",
            "libero_plus",
            "--num-trials",
            "1",
            "--download-strategies",
            "hfd-mirror",
        ]
        with mock.patch.object(sys, "argv", base_argv):
            args = worker.parse_args()
        self.assertTrue(args.no_init_randomization)

        invalid_trials_argv = list(base_argv)
        invalid_trials_argv[invalid_trials_argv.index("--num-trials") + 1] = "2"
        with mock.patch.object(sys, "argv", invalid_trials_argv):
            with self.assertRaises(SystemExit):
                worker.parse_args()
        with mock.patch.object(sys, "argv", [*base_argv, "--task-ids", "0,1"]):
            with self.assertRaises(SystemExit):
                worker.parse_args()


if __name__ == "__main__":
    unittest.main()
