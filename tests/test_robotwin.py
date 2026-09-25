import pathlib
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "libero_eval"))

from robotwin_backend import (  # noqa: E402
    OFFICIAL_EPISODES_PER_TASK,
    OFFICIAL_TASKS,
    _robotwin_environment,
    build_client_command,
    build_protocol_report,
    build_server_command,
    build_worker_slots,
    compare_published_task_results,
    load_published_task_references,
    parse_task_log,
    select_tasks,
    summarize,
)


class TestRoboTwinSelection(unittest.TestCase):
    def test_default_is_the_official_fifty_tasks(self):
        self.assertEqual(select_tasks(None, None), OFFICIAL_TASKS)
        self.assertEqual(len(OFFICIAL_TASKS), 50)

    def test_names_and_indexes_share_one_inventory(self):
        self.assertEqual(select_tasks("lift_pot,hanging_mug", None), OFFICIAL_TASKS[:2])
        self.assertEqual(select_tasks(None, "0,1"), OFFICIAL_TASKS[:2])
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            select_tasks("lift_pot", "0")
        with self.assertRaisesRegex(ValueError, "unknown"):
            select_tasks("not_a_task", None)


class TestRoboTwinProtocol(unittest.TestCase):
    def test_runtime_environment_bypasses_proxy_and_loads_setup_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / ".runtime_library_path").write_text("/conda/lib\n")
            with mock.patch.dict(
                "os.environ",
                {"HTTP_PROXY": "http://proxy", "NO_PROXY": "example.com", "LD_LIBRARY_PATH": "/host/lib"},
                clear=True,
            ):
                env = _robotwin_environment(root, 3)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(env["LD_LIBRARY_PATH"], "/conda/lib:/host/lib")
        self.assertEqual(env["NO_PROXY"], "example.com,localhost,127.0.0.1,0.0.0.0")
        self.assertEqual(env["no_proxy"], env["NO_PROXY"])

    def test_commands_use_isolated_uv_python_paths(self):
        server = build_server_command(pathlib.Path("/lingbot/.venv/bin/python"), pathlib.Path("/model"), 9330, True)
        client = build_client_command(
            pathlib.Path("/robotwin/.venv/bin/python"),
            pathlib.Path("/robotwin/script/client.py"),
            pathlib.Path("/robotwin"),
            "lift_pot",
            "demo_clean",
            9330,
            pathlib.Path("/output"),
        )
        self.assertEqual(server[0], "/lingbot/.venv/bin/python")
        self.assertIn("deploy.lingbot_vla_v2_policy", server)
        self.assertEqual(client[0], "/robotwin/.venv/bin/python")
        self.assertIn("lift_pot", client)

        seen_client = build_client_command(
            pathlib.Path("/robotwin/.venv/bin/python"),
            pathlib.Path("/robotwin/script/client.py"),
            pathlib.Path("/robotwin"),
            "lift_pot",
            "demo_clean",
            9330,
            pathlib.Path("/output"),
            "seen",
        )
        self.assertEqual(seen_client[-2:], ["--instruction_type", "seen"])

    def test_worker_slots_share_one_serial_policy_server_per_gpu(self):
        self.assertEqual(
            build_worker_slots([2, 3], [9200, 9201], 2),
            [(2, 9200), (2, 9200), (3, 9201), (3, 9201)],
        )
        with self.assertRaisesRegex(ValueError, "at least 1"):
            build_worker_slots([2], [9200], 0)

    def test_parses_exact_episode_counts_from_ansi_log(self):
        successes, episodes, rate = parse_task_log("Success rate: \x1b[96m93/100\x1b[0m => \x1b[95m93.0%\x1b[0m")
        self.assertEqual((successes, episodes), (93, OFFICIAL_EPISODES_PER_TASK))
        self.assertEqual(rate, 0.93)

    def test_summary_is_micro_average_and_uses_worker_schema(self):
        raw = {
            "lift_pot": {"status": "ok", "num_successes": 90, "num_trials": 100, "success_rate": 0.9},
            "hanging_mug": {"status": "ok", "num_successes": 80, "num_trials": 100, "success_rate": 0.8},
        }
        summary = summarize(raw, OFFICIAL_TASKS[:2], "demo_clean")
        self.assertEqual(summary["total_success_rate"], 0.85)
        self.assertEqual(summary["suites"]["robotwin_clean"]["episodes"], 200)
        self.assertEqual(summary["tasks"]["lift_pot"]["task_suite_name"], "robotwin_clean")

    def test_reads_and_compares_checkpoint_model_card_task_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            checkpoint = root / "checkpoints" / "step" / "hf_ckpt"
            checkpoint.mkdir(parents=True)
            rows = "\n".join(f"| `{task}` | 90% | 80% |" for task in OFFICIAL_TASKS)
            (root / "README.md").write_text(rows)
            references = load_published_task_references(checkpoint, "demo_clean")

        self.assertEqual(len(references), len(OFFICIAL_TASKS))
        self.assertEqual(references["lift_pot"], 0.9)
        comparison = compare_published_task_results(
            {
                "lift_pot": {
                    "status": "ok",
                    "num_successes": 92,
                    "num_trials": 100,
                    "success_rate": 0.92,
                }
            },
            references,
        )
        self.assertEqual(comparison["tasks_compared"], 1)
        self.assertEqual(comparison["aggregate_deviation_percentage_points"], 2.0)
        self.assertEqual(comparison["exact_task_matches"], 0)

    def test_protocol_completeness_and_published_score_match_are_independent(self):
        results = {
            task: {
                "status": "ok",
                "num_trials": 100,
                "num_successes": 94 if index < 26 else 93,
            }
            for index, task in enumerate(OFFICIAL_TASKS)
        }
        report = build_protocol_report(results, OFFICIAL_TASKS, "demo_clean", [2], 1)

        self.assertTrue(report["official_result"])
        self.assertTrue(report["published_reference"]["exact_aggregate_match"])
        self.assertTrue(report["published_reference"]["statistical_comparison"]["statistically_consistent_at_95pct"])
        self.assertFalse(report["published_reference"]["execution_topology_reported"])
        self.assertEqual(
            report["published_reference"]["upstream_launcher_defaults_at_release"]["policy_servers_total"],
            24,
        )
        self.assertEqual(report["local_execution_topology"]["policy_servers_total"], 1)

        results[OFFICIAL_TASKS[0]]["num_successes"] -= 1
        report = build_protocol_report(results, OFFICIAL_TASKS, "demo_clean", [2], 1)
        self.assertTrue(report["official_result"])
        self.assertFalse(report["published_reference"]["exact_aggregate_match"])

        report = build_protocol_report(results, OFFICIAL_TASKS[:-1], "demo_clean", [2], 1)
        self.assertFalse(report["official_result"])
        self.assertIsNone(report["published_reference"]["exact_aggregate_match"])
        self.assertIsNone(report["published_reference"]["statistical_comparison"])

        report = build_protocol_report(results, OFFICIAL_TASKS, "demo_clean", [2], 1, "seen")
        self.assertEqual(report["instruction_type"], "seen")
        self.assertFalse(report["official_request"])
        self.assertFalse(report["official_result"])

        for row in results.values():
            row["num_successes"] = 85
        report = build_protocol_report(results, OFFICIAL_TASKS, "demo_clean", [2], 1)
        self.assertFalse(report["published_reference"]["statistical_comparison"]["statistically_consistent_at_95pct"])


if __name__ == "__main__":
    unittest.main()
