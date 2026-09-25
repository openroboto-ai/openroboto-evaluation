import json
import pathlib
import sys
import tempfile
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "libero_eval"))

from check_model import CONFIG_SPECS, CheckResult, _orbax_uses_partitioned_write_shapes  # noqa: E402
from robodojo_backend import (  # noqa: E402
    OFFICIAL_SEEDS,
    RunSpec,
    build_client_command,
    build_server_command,
    build_specs,
    parse_dimensions,
    select_tasks,
    summarize,
)


def _run(task, seed, successes, count, score=None):
    score = successes / count if score is None else score
    episodes = [{"layout_id": i, "success": i < successes, "score": score} for i in range(count)]
    return {
        "status": "ok",
        "sim_task": task,
        "report_task": task.removesuffix("_random"),
        "eval_seed": seed,
        "episodes": episodes,
    }


class TestRoboDojoSelection(unittest.TestCase):
    def test_official_inventory_expands_random_halves(self):
        dimensions = parse_dimensions(None)
        tasks = select_tasks(dimensions, None)
        self.assertEqual(len(tasks), 42)
        specs = build_specs(tasks, OFFICIAL_SEEDS)
        self.assertEqual(len(specs), 162)  # (42 base + 12 random) x 3 seeds
        self.assertEqual(sum(spec.sim_task.endswith("_random") for spec in specs), 36)

    def test_dimension_alias_and_task_validation(self):
        self.assertEqual(parse_dimensions("memory,long_horizon"), ("memory", "long-horizon"))
        with self.assertRaisesRegex(ValueError, "unknown task"):
            select_tasks(("memory",), "stack_bowls")

    def test_robodojo_norm_contract_is_dual_arm_14d(self):
        pi05 = CONFIG_SPECS["pi05_robodojo"]
        pi0 = CONFIG_SPECS["pi0_robodojo"]
        self.assertEqual(pi05.asset_id, "arx_x5_sim")
        self.assertEqual(pi05.norm_dims, {"state": 14, "actions": 14})
        self.assertTrue(pi05.use_quantile_norm)
        self.assertFalse(pi0.use_quantile_norm)

    def test_partitioned_orbax_write_shapes_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            params = pathlib.Path(tmp)
            (params / "_sharding").write_text(
                json.dumps({
                    "encoded-name": json.dumps({
                        "sharding_type": "NamedSharding",
                        "partition_spec": [None, "fsdp", None],
                    })
                })
            )
            result = CheckResult(str(params.parent), "pi05_robodojo")
            self.assertTrue(_orbax_uses_partitioned_write_shapes(params, result))
            self.assertEqual(result.errors, [])


class TestRoboDojoCommands(unittest.TestCase):
    def test_server_gets_real_checkpoint_client_gets_safe_label(self):
        root = pathlib.Path("/opt/RoboDojo")
        checkpoint = pathlib.Path("/models/RoboDojo/Pi_05/seed0/59999")
        spec = RunSpec("memory", "cover_blocks", "cover_blocks", 2)
        server = build_server_command(root, "Pi_05", spec, checkpoint, 3, 9123)
        client = build_client_command(root, "Pi_05", spec, "run_eval_pi05_seed0", 3, 9123, 1)
        self.assertIn(str(checkpoint), server)
        self.assertNotIn(str(checkpoint), client)
        self.assertIn("run_eval_pi05_seed0", client)
        self.assertEqual(client[:6], ["conda", "run", "--no-capture-output", "-n", "RoboDojo", "bash"])
        self.assertEqual(client[-2:], ["--eval-num", "1"])


class TestRoboDojoSummary(unittest.TestCase):
    def test_generalization_pairs_25_plus_25_and_macro_averages_dimensions(self):
        raw = {
            "base": _run("stack_bowls", 0, 25, 25, score=1.0),
            "random": _run("stack_bowls_random", 0, 0, 25, score=0.0),
            "precision": _run("fasten_screws", 0, 25, 50, score=0.25),
        }
        summary = summarize(raw, ("stack_bowls", "fasten_screws"), (0,), None)
        gen = summary["task_seed_results"]["stack_bowls:seed0"]
        self.assertEqual(gen["episodes"], 50)
        self.assertEqual(gen["success_rate"], 0.5)
        self.assertEqual(summary["dimensions"]["precision"]["score"], 0.25)
        # Official overview gives every capability dimension equal weight.
        self.assertEqual(summary["total_success_rate"], 0.5)
        self.assertEqual(summary["total_score"], 0.375)

    def test_incomplete_generalization_half_does_not_fill_cell(self):
        raw = {
            "base": _run("stack_bowls", 0, 25, 25),
            "random": _run("stack_bowls_random", 0, 0, 24),
        }
        summary = summarize(raw, ("stack_bowls",), (0,), None)
        self.assertEqual(summary["completed_task_seed_cells"], 0)
        self.assertEqual(summary["task_seed_results"]["stack_bowls:seed0"]["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
