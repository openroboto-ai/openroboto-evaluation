import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build_axis_task_subset import build  # noqa: E402
from axis_runtime import load_manifest, verify_task_payload  # noqa: E402
from verify_axis_release import verify_release  # noqa: E402


class TestAxisTaskSubset(unittest.TestCase):
    def test_subset_preserves_task_semantics_and_records_its_limited_pool(self):
        source = ROOT / "configs/benchmarks/axis_v1.0.yaml"
        original = load_manifest(source)
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "axis_v1.1.json"
            subset = build(source, [502, 501], output, revision="subset-v1", scope="current-task-subset")
            self.assertEqual(subset["tasks"], [task for task in original["tasks"] if task["task_id"] in (501, 502)])
            self.assertEqual(subset["protocol"], original["protocol"])
            self.assertEqual(subset["runtime"], original["runtime"])
            verified = verify_release(output, output.with_name("axis_v1.1-tasks"))
            self.assertEqual(verified["benchmark"], "axis_v1.1")
            self.assertEqual(set(verified["tasks"]), {"501", "502"})
            self.assertEqual(
                subset["subset_provenance"]["source_task_ids"], [task["task_id"] for task in original["tasks"]]
            )
            for spec in subset["tasks"]:
                payload = json.loads((output.with_name("axis_v1.1-tasks") / f"{spec['task_id']}.json").read_text())
                verify_task_payload(payload, spec)
            with self.assertRaises(FileExistsError):
                build(source, [501], output, revision="subset-v1", scope="current-task-subset")

    def test_unavailable_task_or_reused_version_cannot_create_a_fake_extension(self):
        source = ROOT / "configs/benchmarks/axis_v1.0.json"
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "axis_v1.1.json"
            for task_ids in ([], [501, 501], [507]):
                with self.subTest(ids=task_ids), self.assertRaisesRegex(ValueError, "unique subset"):
                    build(source, task_ids, output, revision="subset-v1", scope="current-task-subset")
            with self.assertRaisesRegex(ValueError, "new version"):
                build(source, [501], output.with_name("axis_v1.0.json"), revision="subset-v1", scope="subset")
            self.assertFalse(output.exists())

    def test_release_verifier_rejects_duplicates_and_randomized_base_snapshots(self):
        source = ROOT / "configs/benchmarks/axis_v1.0.json"
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "axis_v1.1.json"
            subset = build(source, [501], output, revision="subset-v1", scope="subset")
            subset["tasks"].append(subset["tasks"][0])
            output.write_text(json.dumps(subset))
            with self.assertRaisesRegex(ValueError, "unique task"):
                verify_release(output, output.with_name("axis_v1.1-tasks"))
            subset["protocol"]["randomization"] = True
            output.write_text(json.dumps(subset))
            with self.assertRaisesRegex(ValueError, "randomized"):
                verify_release(output, output.with_name("axis_v1.1-tasks"))
