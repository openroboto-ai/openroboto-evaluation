"""benchmark_worker/scoring.py 队列字段规整与 env 聚合的单元测试。

回归背景:
- /api/pending-tasks 曾把 env_list 发成 JSON 字符串,直接迭代拆成单字符,
  任务被错报成 "invalid env names" 秒失败(parse_env_list)。
- 后端轮次配置把 env 换成 libero_100(LIBERO-100 = 90+10 的合称,没有独立
  bddl 资产),直接当 suite 执行会 FileNotFoundError,失败报告又因缺 env 被
  后端 400 拒收(expand_env_suites / build_score_payload 按 env 名聚合)。

运行:uv run python -m unittest discover tests
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmark_worker.scoring import (  # noqa: E402
    build_score_payload,
    choose_benchmark,
    expand_env_suites,
    parse_env_list,
    prepare_submit_payload,
)

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_100"]


class TestParseEnvList(unittest.TestCase):
    def test_list_passes_through(self):
        self.assertEqual(parse_env_list(SUITES), SUITES)

    def test_json_string_is_decoded(self):
        # 事故原样输入:旧端点把数组编码成了 JSON 字符串。
        raw = '["libero_spatial", "libero_object", "libero_goal", "libero_100"]'
        self.assertEqual(parse_env_list(raw), SUITES)

    def test_none_and_empty_mean_no_suites(self):
        self.assertEqual(parse_env_list(None), [])
        self.assertEqual(parse_env_list([]), [])
        self.assertEqual(parse_env_list("[]"), [])

    def test_whitespace_entries_are_stripped_and_dropped(self):
        self.assertEqual(parse_env_list([" libero_10 ", "", "  "]), ["libero_10"])

    def test_garbage_string_raises(self):
        with self.assertRaises(ValueError):
            parse_env_list("libero_spatial,libero_object")

    def test_non_list_shapes_raise(self):
        for raw in ({"a": 1}, '{"a": 1}', [1, 2], 42):
            with self.assertRaises(ValueError):
                parse_env_list(raw)


class TestExpandEnvSuites(unittest.TestCase):
    def test_libero_100_expands_to_90_plus_10(self):
        self.assertEqual(
            expand_env_suites(SUITES),
            ["libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10"],
        )

    def test_plain_names_pass_through(self):
        self.assertEqual(
            expand_env_suites(["libero_10", "libero_spatial_lan"]),
            ["libero_10", "libero_spatial_lan"],
        )

    def test_expansion_dedups_preserving_order(self):
        self.assertEqual(expand_env_suites(["libero_10", "libero_100"]), ["libero_10", "libero_90"])

    def test_libero_100_round_runs_on_base_libero_benchmark(self):
        # 回归:libero_100 直接喂 choose_benchmark 会被当扰动 suite 送进
        # libero_pro,prepare 阶段即 FileNotFoundError;展开后全是 base suite。
        self.assertEqual(choose_benchmark(expand_env_suites(SUITES)), "libero")


def _suite_agg(sr, episodes, tasks, failed=0):
    return {"success_rate": sr, "episodes": episodes, "tasks": tasks, "failed_tasks": failed}


def _complete_tasks(suite_counts: dict[str, int], trials: int) -> dict:
    return {
        f"{suite}_task{task_id:04d}": {
            "status": "ok",
            "task_suite_name": suite,
            "task_id": task_id,
            "num_trials": trials,
            "num_successes": 0,
            "success_rate": 0.0,
        }
        for suite, count in suite_counts.items()
        for task_id in range(count)
    }


class TestBuildScorePayloadProfiles(unittest.TestCase):
    """评测 profile 是内容和结构化 score identity 的唯一来源。"""

    def _summary(self):
        suite_counts = {
            "libero_spatial": 10,
            "libero_object": 10,
            "libero_goal": 10,
            "libero_10": 10,
        }
        return {
            "tasks": _complete_tasks(suite_counts, 10),
            "num_trials_per_task": 10,
            "suites": {
                "libero_spatial": _suite_agg(0.8, 100, 10),
                "libero_object": _suite_agg(0.6, 100, 10),
                "libero_goal": _suite_agg(0.7, 100, 10),
                "libero_10": _suite_agg(0.9, 100, 10),
            },
        }

    def test_libero_uses_four_real_base_suite_names(self):
        payload = build_score_payload({"env_list": SUITES}, self._summary(), 10.0, benchmark="libero")
        self.assertEqual(
            [e["env_name"] for e in payload["env_scores"]],
            ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        )
        self.assertTrue(all(e["perturbation"] is None for e in payload["env_scores"]))
        self.assertEqual(payload["env_scores"][1]["base_suite"], "libero_object")
        self.assertTrue(payload["success"])
        self.assertAlmostEqual(payload["total_score"], round((0.8 + 0.6 + 0.7 + 0.9) / 4, 6))

    def test_env_without_results_reported_as_zero_with_error(self):
        summary = self._summary()
        del summary["suites"]["libero_10"]
        payload = build_score_payload({}, summary, 10.0, benchmark="libero")
        by_name = {e["env_name"]: e for e in payload["env_scores"]}
        entry = by_name["libero_10"]
        self.assertEqual((entry["score"], entry["samples"]), (0.0, 0))
        self.assertEqual(entry["error"], "no results for this env")
        self.assertFalse(payload["success"])
        self.assertEqual(payload["total_score"], 0.0)
        self.assertIn("incomplete evaluation", payload["error"])

    def test_libero_plus_total_is_micro_average_over_10030_tasks(self):
        rates = {
            "libero_spatial": 0.5,
            "libero_object": 0.6,
            "libero_goal": 0.7,
            "libero_10": 0.8,
        }
        counts = {
            "libero_spatial": 2402,
            "libero_object": 2518,
            "libero_goal": 2591,
            "libero_10": 2519,
        }
        summary = {
            "tasks": _complete_tasks(counts, 1),
            "num_trials_per_task": 1,
            "suites": {suite: _suite_agg(rates[suite], counts[suite], counts[suite]) for suite in counts},
        }
        payload = build_score_payload({}, summary, 10.0, benchmark="libero_plus")
        expected = sum(rates[suite] * counts[suite] for suite in counts) / 10030

        self.assertEqual(sum(entry["samples"] for entry in payload["env_scores"]), 10030)
        self.assertEqual(payload["total_score"], round(expected, 6))

    def test_custom_1_collapses_16_pro_suites_to_six_backend_names(self):
        suite_names = [
            f"{base}_{dimension}"
            for base in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
            for dimension in ("object", "swap", "lan", "task")
        ]
        rates = {
            "libero_spatial": (0.1, 0.2, 0.3, 0.4),
            "libero_object": (0.5, 0.6, 0.7, 0.8),
            "libero_goal": (0.2, 0.4, 0.6, 0.8),
            "libero_10": (0.9, 0.7, 0.5, 0.3),
        }
        summary = {
            "tasks": _complete_tasks({name: 10 for name in suite_names}, 50),
            "num_trials_per_task": 50,
            "suites": {
                f"{base}_{dimension}": _suite_agg(score, 500, 10)
                for base, scores in rates.items()
                for dimension, score in zip(("object", "swap", "lan", "task"), scores, strict=True)
            },
        }

        standard = build_score_payload({}, summary, 10.0, benchmark="libero_pro")
        custom = build_score_payload({}, summary, 10.0, benchmark="libero_pro_custom_1")

        self.assertEqual(len(standard["env_scores"]), 16)
        self.assertEqual(
            [entry["env_name"] for entry in custom["env_scores"]],
            [
                "libero_spatial",
                "libero_object",
                "libero_goal",
                "libero_10",
                "libero_object_swap",
                "libero_spatial_swap",
            ],
        )
        self.assertEqual(
            [entry["score"] for entry in custom["env_scores"]],
            [0.25, 0.65, 0.5, 0.6, 0.6, 0.2],
        )
        self.assertEqual([entry["samples"] for entry in custom["env_scores"]], [2000] * 4 + [500, 500])
        self.assertAlmostEqual(custom["total_score"], round((0.25 + 0.65 + 0.5 + 0.6 + 0.6 + 0.2) / 6, 6))
        self.assertTrue(all("base_suite" not in entry for entry in custom["env_scores"]))
        self.assertEqual(len(suite_names), 16)  # Custom 1 的实际运行内容没有减少。

    def test_plain_libero_pro_keeps_all_16_structured_scores(self):
        suite_names = [
            f"{base}_{dimension}"
            for base in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
            for dimension in ("object", "swap", "lan", "task")
        ]
        summary = {
            "tasks": _complete_tasks({name: 10 for name in suite_names}, 50),
            "num_trials_per_task": 50,
            "suites": {name: _suite_agg(0.5, 500, 10) for name in suite_names},
        }

        payload = build_score_payload({}, summary, 10.0, benchmark="libero_pro")

        self.assertEqual(len(payload["env_scores"]), 16)
        self.assertTrue(all("base_suite" in entry and "perturbation" in entry for entry in payload["env_scores"]))

    def test_queue_env_list_is_ignored(self):
        payload = build_score_payload({"env_list": ["wrong", "libero_100"]}, self._summary(), 10.0, benchmark="libero")
        self.assertEqual([e["env_name"] for e in payload["env_scores"]][-1], "libero_10")

    def test_97_of_160_tasks_can_never_be_published_as_success(self):
        suite_names = [
            f"{base}_{dimension}"
            for base in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
            for dimension in ("object", "swap", "lan", "task")
        ]
        tasks = _complete_tasks({name: 10 for name in suite_names}, 50)
        tasks = dict(list(tasks.items())[:97])
        summary = {
            "tasks": tasks,
            "num_trials_per_task": 50,
            "suites": {name: _suite_agg(0.5, 500, 10) for name in suite_names},
        }

        payload = build_score_payload({}, summary, 10.0, benchmark="libero_pro_custom_1")

        self.assertFalse(payload["success"])
        self.assertEqual(payload["total_score"], 0.0)
        self.assertIn("completed 97/160 required tasks", payload["error"])

    def test_legacy_queue_hotkey_is_normalized_for_submission(self):
        payload = build_score_payload({"hotkey": "legacy-hotkey"}, None, 10.0, "failed")
        self.assertEqual(payload["miner_hotkey"], "legacy-hotkey")

        payload = build_score_payload(
            {"hotkey": "legacy-hotkey", "miner_hotkey": "canonical-hotkey"}, None, 10.0, "failed"
        )
        self.assertEqual(payload["miner_hotkey"], "canonical-hotkey")


class TestPrepareSubmitPayload(unittest.TestCase):
    def test_omits_optional_task_details_without_mutating_stored_payload(self):
        payload = {
            "success": True,
            "env_scores": [{"env_name": "libero_spatial", "score": 0.5}],
            "total_score": 0.5,
            "error": "",
            "per_task_scores": [{"task_id": "libero_spatial_00", "success_rate": 0.5, "trials": 10}],
        }

        submitted = prepare_submit_payload(payload)

        self.assertEqual(submitted["per_task_scores"], [])
        self.assertEqual(len(payload["per_task_scores"]), 1)
        self.assertEqual(submitted["env_scores"], payload["env_scores"])
        self.assertEqual(submitted["error"], "")

    def test_payload_without_task_details_is_reused(self):
        payload = {"success": False, "per_task_scores": []}
        self.assertIs(prepare_submit_payload(payload), payload)

    def test_old_custom_1_payload_is_collapsed_before_http_submission(self):
        env_scores = [
            {
                "base_suite": base,
                "perturbation": dimension,
                "env_name": f"{base}_{dimension}",
                "score": 0.5,
                "samples": 500,
                "duration_sec": 10.0,
                "error": "",
            }
            for base in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
            for dimension in ("object", "swap", "lan", "task")
        ]
        payload = {
            "benchmark": "libero_pro_custom_1",
            "success": True,
            "env_scores": env_scores,
            "total_score": 0.5,
            "per_task_scores": [],
        }

        submitted = prepare_submit_payload(payload)

        self.assertEqual(len(payload["env_scores"]), 16)
        self.assertEqual(
            [entry["env_name"] for entry in submitted["env_scores"]],
            [
                "libero_spatial",
                "libero_object",
                "libero_goal",
                "libero_10",
                "libero_object_swap",
                "libero_spatial_swap",
            ],
        )
        self.assertEqual(submitted["total_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
