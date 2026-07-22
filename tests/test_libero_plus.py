"""LIBERO-plus 集成的纯逻辑单元测试。

覆盖(评测执行本身由 run_eval 冒烟验证):
  - LIBERO-plus 官方 profile 使用原始 task.language、10,030 tasks × 1 trial;
  - libero_eval/prompt_clean.clean_task_prompt 仅作为非官方消融工具;
  - libero_eval/run_eval.summarize:按扰动维度(task_groups)的汇总口径
    与按 suite 的口径一致,包括失败任务计数。

运行:uv run python -m unittest discover tests
"""

import dataclasses
import pathlib
import sys
import unittest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "libero_eval"))

from prompt_clean import clean_task_prompt  # noqa: E402
from benchmarks import PLUS_SUITE_TASK_COUNTS, PLUS_TOTAL_TASKS, get_benchmark  # noqa: E402
from run_eval import TaskSpec, split_resumable, summarize  # noqa: E402

_BASE = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
_BASE_PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"


class TestCleanTaskPrompt(unittest.TestCase):
    def test_strips_each_perturbation_suffix(self):
        # 每个维度的文件名后缀形态各取一例(真实任务名)。
        for name in [
            f"{_BASE}_table_1",
            f"{_BASE}_tb_12",
            f"{_BASE}_light_02",
            f"{_BASE}_add_10",
            f"{_BASE}_level1_sample1",
            f"{_BASE}_view_0_0_100_0_0_initstate_1",
            f"{_BASE}_view_43_0_100_0_0_initstate_0_noise_38",
        ]:
            junk_language = " ".join(name.split("_"))
            self.assertEqual(clean_task_prompt(name, junk_language), _BASE_PROMPT, name)

    def test_language_tasks_pass_through(self):
        # _language_ 任务的 task.language 已是 fork 从 bddl 解析的 paraphrase。
        paraphrase = "Can you place the black bowl thats between the plate and the ramekin on top of the plate"
        self.assertEqual(clean_task_prompt(f"{_BASE}_language_10", paraphrase), paraphrase)
        # language + view 组合名同样直通(fork 解析 _view_ 前缀对应的 bddl)。
        self.assertEqual(clean_task_prompt(f"{_BASE}_language_10_view_0_0_100_0_0_initstate_0", paraphrase), paraphrase)

    def test_libero100_scene_prefix(self):
        name = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_0"
        self.assertEqual(clean_task_prompt(name, "x"), "turn on the stove and put the moka pot on it")
        # SCENE10 是两位数场景号(上游对此有专门分支)。
        self.assertEqual(
            clean_task_prompt("KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_light_3", "x"),
            "close the top drawer of the cabinet",
        )

    def test_instruction_words_are_not_suffixes(self):
        # 指令里的 table/light 等词后面没有数字,不能被误剥。
        for name, prompt in [
            ("put_the_black_bowl_on_the_table", "put the black bowl on the table"),
            ("turn_on_the_light", "turn on the light"),
        ]:
            self.assertEqual(clean_task_prompt(name, prompt), prompt)


class TestOfficialProtocol(unittest.TestCase):
    def test_default_profile_matches_published_protocol(self):
        bench = get_benchmark("libero_plus")
        counts = dict(PLUS_SUITE_TASK_COUNTS)
        protocol = bench.protocol_request(list(counts), counts, None, 1)

        self.assertEqual(PLUS_TOTAL_TASKS, 10030)
        self.assertEqual(bench.prompt_source, "task")
        self.assertEqual(bench.default_num_trials, 1)
        self.assertTrue(protocol["official_request"])
        self.assertEqual(protocol["scheduled_task_count"], 10030)
        self.assertEqual(protocol["deviations"], [])

    def test_subset_or_trial_override_is_explicitly_non_official(self):
        bench = get_benchmark("libero_plus")
        counts = dict(PLUS_SUITE_TASK_COUNTS)
        protocol = bench.protocol_request(list(counts), counts, [0, 1], 2)

        self.assertFalse(protocol["official_request"])
        self.assertEqual(protocol["scheduled_task_count"], 8)
        self.assertIn("--task-ids selects a development subset", protocol["deviations"])
        self.assertIn("--num-trials must be 1, got 2", protocol["deviations"])

    def test_clean_prompt_cannot_be_labeled_official(self):
        bench = dataclasses.replace(get_benchmark("libero_plus"), prompt_source="task_clean")
        counts = dict(PLUS_SUITE_TASK_COUNTS)
        protocol = bench.protocol_request(list(counts), counts, None, 1)

        self.assertFalse(protocol["official_request"])
        self.assertIn("prompt_source must be 'task', got 'task_clean'", protocol["deviations"])


class TestSummarizeDimensions(unittest.TestCase):
    def _results(self):
        return {
            "libero_spatial_task00": {
                "status": "ok",
                "task_suite_name": "libero_spatial",
                "task_id": 0,
                "num_trials": 1,
                "num_successes": 1,
            },
            "libero_spatial_task01": {
                "status": "ok",
                "task_suite_name": "libero_spatial",
                "task_id": 1,
                "num_trials": 1,
                "num_successes": 0,
            },
            "libero_goal_task00": {
                "status": "failed",
                "task_suite_name": "libero_goal",
                "task_id": 0,
            },
        }

    def test_dimension_rollup(self):
        groups = {
            "libero_spatial": {0: "Camera Viewpoints", 1: "Sensor Noise"},
            "libero_goal": {0: "Camera Viewpoints"},
        }
        summary = summarize(self._results(), {}, task_groups=groups)
        dims = summary["dimensions"]
        self.assertEqual(
            dims["Camera Viewpoints"],
            {"tasks": 2, "failed_tasks": 1, "episodes": 1, "successes": 1, "success_rate": 1.0},
        )
        self.assertEqual(
            dims["Sensor Noise"],
            {"tasks": 1, "failed_tasks": 0, "episodes": 1, "successes": 0, "success_rate": 0.0},
        )
        # 维度汇总不改变按 suite 的口径与总数。
        self.assertEqual(summary["total_episodes"], 2)
        self.assertEqual(summary["suites"]["libero_goal"]["failed_tasks"], 1)

    def test_tasks_outside_mapping_are_skipped(self):
        groups = {"libero_spatial": {0: "Camera Viewpoints"}}  # task 1 与 libero_goal 未映射
        dims = summarize(self._results(), {}, task_groups=groups)["dimensions"]
        self.assertEqual(list(dims), ["Camera Viewpoints"])
        self.assertEqual(dims["Camera Viewpoints"]["tasks"], 1)

    def test_no_groups_no_dimensions_key(self):
        self.assertNotIn("dimensions", summarize(self._results(), {}))


class TestSplitResumable(unittest.TestCase):
    def test_done_broken_and_missing(self):
        import json
        import tempfile

        specs = [TaskSpec("libero_spatial", i, 220) for i in range(3)]
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d)
            (out / "results").mkdir()
            done = {"task_suite_name": "libero_spatial", "task_id": 0, "num_trials": 1, "num_successes": 1}
            (out / "results" / f"{specs[0].name}.json").write_text(json.dumps(done))
            (out / "results" / f"{specs[1].name}.json").write_text("{trunca")  # 进程被杀时写了一半
            todo, results = split_resumable(specs, out)
        self.assertEqual([sp.task_id for sp in todo], [1, 2])
        entry = results[specs[0].name]
        self.assertEqual((entry["status"], entry["resumed"], entry["num_successes"]), ("ok", True, 1))


if __name__ == "__main__":
    unittest.main()
