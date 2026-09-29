import copy
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_axis_results import _failed_conditions, analyze  # noqa: E402


class TestAxisAnalysis(unittest.TestCase):
    def test_passed_or_branch_does_not_create_a_spurious_failure_category(self):
        checker = {
            "checker_type": "CompositeChecker",
            "operator": "AND",
            "passed": False,
            "sub_results": [
                {
                    "checker_type": "CompositeChecker",
                    "operator": "OR",
                    "passed": True,
                    "sub_results": [
                        {"checker_type": "DirectedRotationChecker", "passed": True},
                        {"checker_type": "DirectedRotationChecker", "passed": False, "reason": "missing orientation"},
                    ],
                },
                {"checker_type": "GripperOpenChecker", "passed": False},
            ],
        }
        self.assertEqual(_failed_conditions(checker), {"GripperOpenChecker"})

    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "configs" / "benchmarks" / "axis_v1.0.json").read_text())
        cls.task_ids = [task["task_id"] for task in cls.manifest["tasks"]]

    def _summary(self, successful: set[int]) -> dict:
        return {
            "benchmark": "axis_v1.0",
            "dry_run": False,
            "protocol_revision": self.manifest["protocol_revision"],
            "randomization": False,
            "policy_seed": 7,
            "num_trials_per_task": 10,
            "replan_steps": 5,
            "gripper_mode": "continuous",
            "policy_samples": 1,
            "sample_reduction": "mean",
            "renderer_backend": "osmesa",
            "tasks": {
                str(task_id): {
                    "status": "ok",
                    "task_id": task_id,
                    "num_trials": 10,
                    "num_successes": 10 if task_id in successful else 0,
                    "success_rate": 1.0 if task_id in successful else 0.0,
                }
                for task_id in self.task_ids
            },
        }

    def test_complete_runs_produce_difficulty_separation_and_subset(self):
        report = analyze(
            [
                ("weak", self._summary(set())),
                ("middle", self._summary(set(self.task_ids[:15]))),
                ("strong", self._summary(set(self.task_ids))),
            ],
            self.manifest,
        )

        self.assertEqual(report["model_count"], 3)
        self.assertEqual(report["model_statistics"][1]["success_rate"], 0.5)
        self.assertEqual(report["task_statistics"][0]["model_separation"], 1.0)
        self.assertFalse(report["task_statistics"][0]["floor_effect"])
        self.assertTrue(report["recommended_task_ids"])
        self.assertFalse(report["warnings"])

    def test_two_models_are_insufficient_for_correlation_selection(self):
        report = analyze(
            [("weak", self._summary(set())), ("strong", self._summary(set(self.task_ids)))],
            self.manifest,
        )

        self.assertEqual(report["recommended_task_ids"], [])
        self.assertIn("At least three", report["warnings"][0])
        self.assertTrue(all(row["pearson"] is None for row in report["task_correlations"]))

    def test_shared_manifest_does_not_allow_mixed_effective_protocols(self):
        for field, changed in (
            ("policy_seed", 8),
            ("num_trials_per_task", 20),
            ("replan_steps", 10),
            ("gripper_mode", "binary"),
            ("policy_samples", 5),
            ("sample_reduction", "medoid"),
            ("renderer_backend", "egl"),
        ):
            with self.subTest(field=field):
                baseline = self._summary(set())
                other = self._summary(set(self.task_ids))
                other[field] = changed
                with self.assertRaisesRegex(ValueError, f"matching {field}"):
                    analyze([("baseline", baseline), ("other", other)], self.manifest)

    def test_missing_effective_settings_cannot_certify_multi_model_comparison(self):
        for label_index in (0, 1):
            for unknown in (None, "missing"):
                with self.subTest(label_index=label_index, unknown=unknown):
                    summaries = [("weak", self._summary(set())), ("strong", self._summary(set(self.task_ids)))]
                    if unknown == "missing":
                        summaries[label_index][1].pop("replan_steps")
                    else:
                        summaries[label_index][1]["replan_steps"] = None
                    with self.assertRaisesRegex(ValueError, "explicit replan_steps"):
                        analyze(summaries, self.manifest)

    def test_single_legacy_summary_remains_descriptive(self):
        summary = self._summary({501})
        summary.pop("replan_steps")
        report = analyze([("legacy", summary)], self.manifest)
        self.assertEqual(report["model_statistics"][0]["successes"], 10)
        self.assertIsNone(report["common_recorded_protocol"])

    def test_matched_protocol_is_recorded_even_when_model_scores_differ(self):
        report = analyze(
            [("weak", self._summary(set())), ("strong", self._summary(set(self.task_ids)))],
            self.manifest,
        )
        self.assertEqual(report["common_recorded_protocol"]["replan_steps"], 5)
        self.assertEqual(report["common_recorded_protocol"]["policy_seed"], 7)
        self.assertEqual(report["task_statistics"][0]["model_separation"], 1.0)

    def test_randomized_comparison_requires_the_same_recorded_environment_seed(self):
        manifest = copy.deepcopy(self.manifest)
        manifest.update(name="axis_v99.0", protocol_revision="test_randomized_v1")
        manifest["protocol"]["randomization"] = True
        summaries = [("weak", self._summary(set())), ("strong", self._summary(set(self.task_ids)))]
        for _, summary in summaries:
            summary.update(
                benchmark=manifest["name"],
                randomization=True,
                protocol_revision=manifest["protocol_revision"],
                randomization_seed=0,
            )
        self.assertEqual(analyze(summaries, manifest)["common_recorded_protocol"]["randomization_seed"], 0)
        summaries[1][1]["randomization_seed"] = 1
        with self.assertRaisesRegex(ValueError, "matching randomization_seed"):
            analyze(summaries, manifest)

    def test_numerical_runtime_cannot_be_mixed_with_legacy_or_other_runtime(self):
        baseline = self._summary(set())
        baseline["numerical_runtime"] = {"policy": "cache-independent-v1"}
        other = self._summary(set(self.task_ids))
        for legacy in (None, {"policy": "cached-v0"}):
            other["numerical_runtime"] = legacy
            for summaries in ([("a", baseline), ("b", other)], [("b", other), ("a", baseline)]):
                with self.subTest(runtime=legacy), self.assertRaisesRegex(ValueError, "numerical_runtime"):
                    analyze(summaries, self.manifest)
        other["numerical_runtime"] = dict(baseline["numerical_runtime"])
        report = analyze([("a", baseline), ("b", other)], self.manifest)
        self.assertEqual(report["common_recorded_protocol"]["numerical_runtime"], baseline["numerical_runtime"])

    def test_incomplete_or_randomized_run_is_rejected(self):
        incomplete = self._summary(set())
        incomplete["tasks"].pop(str(self.task_ids[-1]))
        with self.assertRaisesRegex(ValueError, "task ids do not match"):
            analyze([("bad", incomplete)], self.manifest)

        randomized = self._summary(set())
        randomized["randomization"] = True
        with self.assertRaisesRegex(ValueError, "randomization"):
            analyze([("bad", randomized)], self.manifest)

    def test_randomized_version_and_terminal_failures_are_analyzed(self):
        manifest = copy.deepcopy(self.manifest)
        manifest.update(name="axis_v99.0", protocol_revision="test_randomized_v1")
        manifest["protocol"]["randomization"] = True
        summary = self._summary(set())
        summary.update(benchmark=manifest["name"], randomization=True, protocol_revision=manifest["protocol_revision"])
        for task in summary["tasks"].values():
            task["episodes"] = [
                {
                    "success": False,
                    "error": None,
                    "checker": {
                        "checker_type": "CompositeChecker",
                        "passed": False,
                        "sub_results": [
                            {"checker_type": "RelativeCylinderChecker", "passed": False},
                            {"checker_type": "GripperOpenChecker", "passed": True},
                        ],
                    },
                }
                for _ in range(10)
            ]
        report = analyze([("demo", summary)], manifest)
        self.assertEqual(report["benchmark"], "axis_v99.0")
        self.assertTrue(report["randomization"])
        failure = report["task_statistics"][0]["model_failures"]["demo"]
        self.assertEqual(failure["goal_not_reached"], 10)
        self.assertEqual(failure["failed_terminal_conditions"], {"RelativeCylinderChecker": 10})
        summary["protocol_revision"] = "different"
        with self.assertRaisesRegex(ValueError, "protocol_revision"):
            analyze([("wrong", summary)], manifest)

    def test_infra_errors_and_inconsistent_episode_totals_are_rejected(self):
        summary = self._summary(set())
        record = summary["tasks"][str(self.task_ids[0])]
        record["episodes"] = [{"success": False, "error": None} for _ in range(10)]
        record["episodes"][0]["error"] = "policy server disconnected"
        with self.assertRaisesRegex(ValueError, "infrastructure error"):
            analyze([("bad", summary)], self.manifest)
        record["episodes"][0] = {"success": True, "error": None}
        with self.assertRaisesRegex(ValueError, "episode successes"):
            analyze([("bad", summary)], self.manifest)

    def test_expanded_manifest_identity_and_hash_are_preserved(self):
        manifest = {**self.manifest, "name": "axis_v99.1", "protocol_revision": "extension-v1"}
        summary = self._summary(set())
        summary.update(benchmark="axis_v99.1", protocol_revision="extension-v1")
        self.assertEqual(analyze([("demo", summary)], manifest)["benchmark"], "axis_v99.1")
        summary["manifest_canonical_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            analyze([("bad", summary)], manifest)


if __name__ == "__main__":
    unittest.main()
