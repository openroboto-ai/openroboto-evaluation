"""Init states 随机化(50/50 混合 + 队列 seed 派生)的单元测试。

覆盖四块纯逻辑(评测执行本身由 run_eval 冒烟验证):
  - libero_eval/init_mix.mix_counts:官方/重采样各半的 trial 分配;
  - libero_eval/init_mix.derive_task_seed:seed → 每 task seed 的公开派生
    函数,测试向量与 prototype 仓库(scripts/verify_init_seed.py、
    docs/SEED_GENERATION.md)共享,两边必须 bitwise 一致;
  - benchmark_worker/worker.select_init_seed:从队列条目选取 init seed,
    缺失/非法回退 None(纯官方评测);
  - benchmark_worker/scoring.build_score_payload:init_seed 随 payload 公布。

运行:uv run python -m unittest discover tests
"""

import json
import pathlib
import sys
import tempfile
import unittest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "libero_eval"))

from benchmark_worker.scoring import build_score_payload  # noqa: E402
from benchmark_worker.worker import select_init_seed  # noqa: E402
from init_mix import derive_task_seed, mix_counts  # noqa: E402
from run_eval import _count_completed_seeded_tasks, _init_task_shards, _required_seeded_inits  # noqa: E402

# 与 prototype 仓库共享的测试向量(scripts/verify_init_seed.py --self-test、
# docs/SEED_GENERATION.md"从 seed 到评测 init states"一节)。改动任何一边的
# 派生实现都必须让这些常量继续成立。
SHARED_VECTORS = [
    (
        3655389442,
        "libero_10_lan",
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        4058840980,
    ),
    (
        3655389442,
        "libero_spatial_object",
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        2968466667,
    ),
    (0, "libero_goal_swap", "open_the_middle_drawer_of_the_cabinet", 536458574),
    (2**32 - 1, "libero_object_task", "pick_up_the_butter_and_place_it_in_the_basket", 2668530624),
]


class TestMixCounts(unittest.TestCase):
    def test_even_split(self):
        self.assertEqual(mix_counts(10, 50, 50), (5, 5))

    def test_odd_split_favours_official(self):
        self.assertEqual(mix_counts(11, 50, 50), (6, 5))
        self.assertEqual(mix_counts(1, 50, 50), (1, 0))

    def test_seeded_side_short_official_absorbs(self):
        self.assertEqual(mix_counts(10, 50, 3), (7, 3))

    def test_official_side_short_seeded_absorbs(self):
        self.assertEqual(mix_counts(10, 3, 50), (3, 7))

    def test_both_sides_short_total_clamped(self):
        self.assertEqual(mix_counts(10, 3, 2), (3, 2))

    def test_zero_and_negative_trials(self):
        self.assertEqual(mix_counts(0, 50, 50), (0, 0))
        self.assertEqual(mix_counts(-1, 50, 50), (0, 0))


class TestDeriveTaskSeed(unittest.TestCase):
    def test_shared_vectors(self):
        for base_seed, suite, task_name, expected in SHARED_VECTORS:
            self.assertEqual(derive_task_seed(base_seed, suite, task_name), expected)

    def test_output_in_numpy_seed_range(self):
        for base_seed, suite, task_name, _ in SHARED_VECTORS:
            seed = derive_task_seed(base_seed, suite, task_name)
            self.assertTrue(0 <= seed < 2**32)


class TestSelectInitSeed(unittest.TestCase):
    def test_valid_seed_passes_through(self):
        self.assertEqual(select_init_seed({"task_id": "t", "seed": 3655389442}), 3655389442)

    def test_uint32_bounds_are_valid(self):
        self.assertEqual(select_init_seed({"task_id": "t", "seed": 0}), 0)
        self.assertEqual(select_init_seed({"task_id": "t", "seed": 2**32 - 1}), 2**32 - 1)

    def test_numeric_string_coerced(self):
        self.assertEqual(select_init_seed({"task_id": "t", "seed": "12345"}), 12345)

    def test_missing_seed_falls_back_to_none(self):
        with self.assertLogs("benchmark_worker", level="WARNING"):
            self.assertIsNone(select_init_seed({"task_id": "t"}))

    def test_invalid_seed_falls_back_to_none(self):
        for bad in ("not-a-number", -1, 2**32, 3.14, None):
            with self.assertLogs("benchmark_worker", level="WARNING"):
                self.assertIsNone(select_init_seed({"task_id": "t", "seed": bad}))


class TestInitProgress(unittest.TestCase):
    def test_counts_manifest_tasks_across_requested_suites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for suite, task_names in (("suite_a", ["a", "b"]), ("suite_b", ["c"])):
                suite_dir = root / suite
                suite_dir.mkdir()
                (suite_dir / "manifest.json").write_text(
                    json.dumps({"tasks": {name: {} for name in task_names}})
                )

            self.assertEqual(_count_completed_seeded_tasks(root, ["suite_a", "suite_b"]), 3)
            self.assertEqual(_count_completed_seeded_tasks(root, ["suite_a"]), 2)

    def test_ignores_missing_or_partially_written_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            suite_dir = root / "suite_a"
            suite_dir.mkdir()
            (suite_dir / "manifest.json").write_text('{"tasks":')

            self.assertEqual(_count_completed_seeded_tasks(root, ["suite_a", "suite_missing"]), 0)


class TestInitGenerationSizing(unittest.TestCase):
    def test_generates_only_the_seeded_half(self):
        self.assertEqual(_required_seeded_inits(1), 0)
        self.assertEqual(_required_seeded_inits(10), 5)
        self.assertEqual(_required_seeded_inits(11), 5)
        self.assertEqual(_required_seeded_inits(50), 25)

    def test_preserves_requested_total_above_official_capacity(self):
        self.assertEqual(_required_seeded_inits(100), 50)
        self.assertEqual(_required_seeded_inits(101), 51)
        self.assertEqual(_required_seeded_inits(200), 150)

    def test_task_sharding_fills_32_workers_for_base_libero(self):
        suites = [f"suite_{i}" for i in range(4)]
        suite_n_tasks = {suite: 10 for suite in suites}
        shards = _init_task_shards(suites, suite_n_tasks, list(range(8)), workers_per_gpu=4)

        assigned = [task for _, _, shard in shards for task in shard]
        expected = [(suite, task_id) for suite in suites for task_id in range(10)]
        self.assertEqual(len(shards), 32)
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), set(expected))
        self.assertTrue(all(1 <= len(shard) <= 2 for _, _, shard in shards))

    def test_task_sharding_balances_libero_pro(self):
        suites = [f"suite_{i}" for i in range(16)]
        suite_n_tasks = {suite: 10 for suite in suites}
        shards = _init_task_shards(suites, suite_n_tasks, list(range(8)), workers_per_gpu=4)

        self.assertEqual(len(shards), 32)
        self.assertTrue(all(len(shard) == 5 for _, _, shard in shards))

    def test_rejects_non_positive_worker_count(self):
        with self.assertRaises(ValueError):
            _init_task_shards(["suite"], {"suite": 10}, [0], workers_per_gpu=0)


class TestPayloadPublishesSeed(unittest.TestCase):
    def test_init_seed_in_payload(self):
        payload = build_score_payload({"round_num": 5}, None, 1.0, init_seed=12345)
        self.assertEqual(payload["init_seed"], 12345)

    def test_init_seed_none_when_disabled(self):
        payload = build_score_payload({"round_num": 5}, None, 1.0)
        self.assertIsNone(payload["init_seed"])


if __name__ == "__main__":
    unittest.main()
