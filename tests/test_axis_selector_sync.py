"""The validator bridge must preserve old execution semantics and freeze new IDs."""

import copy
import json
import pathlib
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_backend import apply_manifest_defaults
from axis_runtime import AXIS_V1_CONFIG_PATH, canonical_json_sha256, load_manifest, task_specs
from export_axis_task_docs import export_docs
from sync_axis_benchmark import sync
from verify_axis_release import verify_release


@pytest.fixture
def inputs(tmp_path):
    # A local selector contract fixture, not a copy of the external algorithm.
    selector = tmp_path / "selector"
    selector.mkdir()
    baseline = load_manifest(AXIS_V1_CONFIG_PATH)
    (selector / "baseline.json").write_text(json.dumps(list(task_specs(baseline))))
    (selector / "selector.py").write_text("""
import hashlib
import json
import pathlib
import sys
ALGORITHM = "test-selector-contract"
IDS = [501, 502, 503, 504, 505, 506, 514, 757] + list(range(5001, 5036))
BASELINE = json.loads(pathlib.Path(__file__).with_name("baseline.json").read_text())
def load_pool():
    return {"tasks": [{"task_id": i, "task_group": str(i)} for i in IDS]}
def validate_pool(pool):
    return {t["task_id"]: t for t in pool["tasks"]}
def pool_fingerprint(pool):
    return hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest()
def normalize_seed(seed):
    if isinstance(seed, str):
        seed = int(seed, 16)
    if type(seed) is not int or not 0 <= seed < 2**256:
        raise ValueError("invalid seed")
    return f"{seed:064x}"
def draw(seed, current_count, selected_ids, pool):
    if (current_count != len(selected_ids) or not set(BASELINE) <= set(selected_ids)
            or not set(selected_ids) <= set(IDS + BASELINE)):
        raise ValueError("invalid selector history")
    remaining = [i for i in IDS if i not in selected_ids]
    if len(remaining) < 10:
        raise ValueError("pool exhausted")
    return sorted(remaining, key=lambda i: hashlib.sha256((normalize_seed(seed) + str(i)).encode()).digest())[:10]
if __name__ == "__main__":
    print(json.dumps(draw(**json.load(sys.stdin), pool=load_pool())))
""")
    source = {**copy.deepcopy(baseline), "name": "axis-v90.0", "tasks": []}
    source.pop("task_snapshot_root")
    template = next(task for task in baseline["tasks"] if task["task_id"] == 501)
    raw = json.loads((AXIS_V1_CONFIG_PATH.parent / baseline["task_snapshot_root"] / "501.json").read_text())
    snapshots = tmp_path / "axis-v90.0-tasks"
    snapshots.mkdir()
    for task_id in [501, 502, 503, 504, 505, 506, 514, 757, *range(5001, 5036)]:
        source["tasks"].append({**template, "task_id": task_id})
        (snapshots / f"{task_id}.json").write_text(json.dumps({**raw, "id": task_id}))
    manifest = tmp_path / "axis-v90.0.json"
    manifest.write_text(json.dumps(source))
    return {"selector_root": selector, "runtime_pool": manifest, "previous": AXIS_V1_CONFIG_PATH, "seed": "0x1234"}


def test_three_rounds_preserve_old_tasks_and_use_complete_history(inputs, tmp_path):
    previous = load_manifest(inputs["previous"])
    initial = copy.deepcopy(previous)
    legacy = sorted(set(task_specs(initial)) - {501, 502, 503, 504, 505, 506, 514, 757})
    prior_receipt = None
    for index, expected_count in enumerate((40, 50, 60), start=1):
        output = tmp_path / f"round{index}"
        receipt = sync(**inputs, output=output)
        path = output / f"axis_v1.{index}.yaml"
        current = load_manifest(path)
        assert len(current["tasks"]) == expected_count
        for filename in ("README.md", "readme_zh.md"):
            document = (output / "task-docs" / filename).read_text()
            assert f"{expected_count}" in document
            assert document.count("https://hub.axisrobotics.ai/explorer-1/task?id=") == expected_count
            for task_id in receipt["selected_task_ids"]:
                assert f"[Hub](https://hub.axisrobotics.ai/explorer-1/task?id={task_id})" in document
            assert str(tmp_path) not in document
            assert "api.axis-labs.ai" not in document
            assert "PREVIEW" in document or "尚未公布" in document
        assert current["tasks"][: len(previous["tasks"])] == previous["tasks"]
        assert current["runtime"] == initial["runtime"]
        assert current["protocol"] == initial["protocol"]
        assert current["policy_seed"] == initial["policy_seed"]
        assert receipt["retained_legacy_task_ids"] == legacy
        assert receipt["selector_request"]["current_count"] == expected_count - 10
        assert receipt["selector_request"]["selected_ids"] == list(task_specs(previous))
        assert len(set(receipt["selected_task_ids"])) == expected_count
        assert not set(receipt["added_task_ids"]) & set(task_specs(previous))
        assert receipt["previous_receipt_sha256"] == (canonical_json_sha256(prior_receipt) if prior_receipt else None)
        for task in previous["tasks"]:
            filename = f"{task['task_id']}.json"
            old = inputs["previous"].parent / previous["task_snapshot_root"] / filename
            new = output / current["task_snapshot_root"] / filename
            assert new.read_bytes() == old.read_bytes()
        assert verify_release(path, output / current["task_snapshot_root"]) == receipt["configuration_verification"]
        previous, prior_receipt = current, receipt
        inputs["previous"] = path
    with pytest.raises(ValueError, match="pool exhausted"):
        sync(**inputs, output=tmp_path / "exhausted")
    assert not (tmp_path / "exhausted").exists()


def test_reproducible_relocatable_bundle_uses_existing_evaluator(inputs, tmp_path):
    first = sync(**inputs, output=tmp_path / "first")
    assert first == sync(**inputs, output=tmp_path / "second")
    moved = tmp_path / "moved"
    shutil.move(tmp_path / "first", moved)
    shutil.rmtree(inputs["runtime_pool"].with_name("axis-v90.0-tasks"))
    manifest = moved / "axis_v1.1.yaml"
    args = SimpleNamespace(
        benchmark="axis", axis_manifest=str(manifest), axis_replan_steps=None, seed=None, config=None
    )
    apply_manifest_defaults(args)
    assert args.axis_loaded_manifest["name"] == "axis_v1.1"
    assert len(args.axis_loaded_manifest["tasks"]) == 40
    assert args.seed == 20260907
    assert args.axis_replan_steps == 10
    assert args.config == "pi05_axis_joint"
    assert verify_release(manifest, moved / "axis_v1.1-tasks") == first["configuration_verification"]


@pytest.mark.parametrize(
    "mutation", ["code", "pool", "manifest", "source_manifest", "payload", "receipt", "seed", "history"]
)
def test_continuation_rejects_drift_and_tampering(inputs, tmp_path, mutation):
    first = tmp_path / "first"
    receipt = sync(**inputs, output=first)
    inputs["previous"] = first / "axis_v1.1.yaml"
    if mutation == "code":
        path = inputs["selector_root"] / "selector.py"
        path.write_text(path.read_text() + "\n# changed version\n")
    elif mutation == "pool":
        path = inputs["runtime_pool"]
        data = json.loads(path.read_text())
        data["description"] = "changed pool"
        path.write_text(json.dumps(data))
    elif mutation == "manifest":
        path = inputs["previous"]
        data = yaml.safe_load(path.read_text())
        data["policy_seed"] += 1
        path.write_text(yaml.safe_dump(data))
    elif mutation == "source_manifest":
        path = first / "axis_v1.1.json"
        data = json.loads(path.read_text())
        data["tasks"][0]["instruction"] = "Modified source"
        path.write_text(json.dumps(data))
    elif mutation == "payload":
        path = first / "axis_v1.1-tasks/501.json"
        data = json.loads(path.read_text())
        data["mjcf_xml"] += "<!-- drift -->"
        path.write_text(json.dumps(data))
    else:
        if mutation == "receipt":
            receipt["added_task_ids"].reverse()
        elif mutation == "history":
            receipt["selector_request"]["selected_ids"] = receipt["previous_public_task_ids"]
            receipt["selector_request"]["current_count"] = len(receipt["previous_public_task_ids"])
        else:
            receipt["selector_request"]["seed"] = "0x5678"
        (first / "selection.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        sync(**inputs, output=tmp_path / "next")
    assert not (tmp_path / "next").exists()


@pytest.mark.parametrize("mutation", ["missing_spec", "missing_payload", "payload_drift", "protocol"])
def test_invalid_runtime_data_never_shrinks_pool_or_redraws(inputs, tmp_path, mutation):
    reference = sync(**inputs, output=tmp_path / "reference")
    task_id = reference["added_task_ids"][0]
    source_path = inputs["runtime_pool"]
    source = load_manifest(source_path)
    if mutation == "missing_spec":
        source["tasks"] = [task for task in source["tasks"] if task["task_id"] != task_id]
    elif mutation == "protocol":
        source["protocol"]["max_control_steps_per_trial"] = 200
    else:
        path = source_path.with_name("axis-v90.0-tasks") / f"{task_id}.json"
        if mutation == "missing_payload":
            path.unlink()
        else:
            data = json.loads(path.read_text())
            data["checker_config"] = {}
            path.write_text(json.dumps(data))
    source_path.write_text(json.dumps(source))
    with pytest.raises((ValueError, FileNotFoundError)):
        sync(**inputs, output=tmp_path / "broken")
    assert not (tmp_path / "broken").exists()


def test_rerun_cannot_overwrite_and_invalid_seed_is_rejected(inputs, tmp_path):
    output = tmp_path / "first"
    receipt = sync(**inputs, output=output)
    with pytest.raises(FileExistsError):
        sync(**inputs, output=output)
    assert json.loads((output / "selection.json").read_text()) == receipt
    with pytest.raises(ValueError, match="invalid seed"):
        sync(**{**inputs, "seed": -1}, output=tmp_path / "badseed")


def test_cli_from_outside_validator_returns_next_benchmark_file(inputs, tmp_path):
    request = {
        "seed": inputs["seed"],
        "current_count": 30,
        "selected_ids": list(task_specs(load_manifest(inputs["previous"]))),
    }
    selection = tmp_path / "new_ids.json"
    selected = subprocess.run(
        [sys.executable, str(inputs["selector_root"] / "selector.py")],
        input=json.dumps(request),
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    selection.write_text(selected.stdout)
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/sync_axis_benchmark.py"),
            "--selector-root",
            str(inputs["selector_root"]),
            "--runtime-pool",
            str(inputs["runtime_pool"]),
            "--previous",
            str(inputs["previous"]),
            "--seed",
            "4660",
            "--selection",
            str(selection),
            "--output",
            str(tmp_path / "cli"),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(process.stdout)
    assert result["benchmark"] == "axis_v1.1"
    assert result["tasks"] == 40
    assert result["added_task_ids"] == json.loads(selected.stdout)
    assert pathlib.Path(result["manifest"]).is_file()
    assert pathlib.Path(result["manifest"]).suffix == ".yaml"
    assert pathlib.Path(result["receipt"]).is_file()
    assert (pathlib.Path(result["task_docs"]) / "README.md").is_file()


def test_selector_output_handoff_and_replay_match_over_two_rounds(inputs, tmp_path):
    for count in (30, 40):
        request = {
            "seed": inputs["seed"],
            "current_count": count,
            "selected_ids": list(task_specs(load_manifest(inputs["previous"]))),
        }
        process = subprocess.run(
            [sys.executable, str(inputs["selector_root"] / "selector.py")],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=True,
            cwd=tmp_path,
        )
        selection = tmp_path / f"selection-{count}.json"
        selection.write_text(process.stdout)
        output = tmp_path / f"handoff-{count}"
        receipt = sync(**inputs, selection=selection, output=output)
        assert receipt == sync(**inputs, output=tmp_path / f"replay-{count}")
        assert receipt["selector_request"]["selected_ids"] == request["selected_ids"]
        assert receipt["added_task_ids"] == json.loads(process.stdout)
        assert receipt["selected_task_ids"] == request["selected_ids"] + json.loads(process.stdout)
        inputs["previous"] = output / f"{receipt['benchmark']}.yaml"


@pytest.mark.parametrize(
    "mutation", ["reordered", "duplicate", "unknown", "old_task", "bool", "short", "object", "json"]
)
def test_imported_selection_must_match_exact_replay(inputs, tmp_path, mutation):
    reference = sync(**inputs, output=tmp_path / "reference")
    added = reference["added_task_ids"].copy()
    if mutation == "reordered":
        added.reverse()
    elif mutation == "duplicate":
        added[0] = added[1]
    elif mutation == "unknown":
        added[0] = 999999
    elif mutation == "old_task":
        added[0] = 501
    elif mutation == "bool":
        added[0] = True
    elif mutation == "short":
        added.pop()
    elif mutation == "object":
        added = {"added_task_ids": added}
    selection = tmp_path / "new_ids.json"
    selection.write_text("invalid json" if mutation == "json" else json.dumps(added))
    with pytest.raises(ValueError):
        sync(**inputs, selection=selection, output=tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_task_docs_preserve_names_and_link_checks_without_publishing_internal_details(inputs, tmp_path):
    data = inputs["selector_root"] / "data"
    data.mkdir()
    (data / "hub_task_links.json").write_text(
        json.dumps({
            "checked_at": "2026-09-23",
            "tasks": [
                {"task_id": 22, "status": "verified", "name": "Grab Can"},
                {"task_id": 31, "status": "verified", "name": "A different task"},
            ],
        })
    )
    first, second = tmp_path / "first-docs", tmp_path / "second-docs"
    for output in (first, second):
        export_docs(
            manifest_path=AXIS_V1_CONFIG_PATH, selector_root=inputs["selector_root"], output=output, status="baseline"
        )
    en = (first / "README.md").read_text()
    zh = (first / "readme_zh.md").read_text()
    assert en.startswith("# axis_v1.0\n")
    assert "抓起易拉罐 / Grab Can" in zh
    assert "[Hub](https://hub.axisrobotics.ai/explorer-1/task?id=22) |" in en
    assert "id=31) · Hub title: A different task" in en
    assert "id=501) · not verified" in en
    for forbidden in (
        "Added this round",
        "Evaluation protocol",
        "Annotations and recorded demonstrations",
        "Success checks",
        "Validation evidence",
        "Provenance and updates",
        "Hz",
        "manifest",
        "Not recorded",
        "seed",
    ):
        assert forbidden not in en
    for filename in ("README.md", "readme_zh.md"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    with pytest.raises(FileExistsError):
        export_docs(manifest_path=AXIS_V1_CONFIG_PATH, selector_root=inputs["selector_root"], output=first)


def test_task_docs_never_publish_internal_evidence(inputs, tmp_path):
    evidence = inputs["selector_root"] / "evidence"
    evidence.mkdir()
    path = evidence / "task_set_validation.json"
    path.write_text(json.dumps({"axis_v1.1": {"summary_en": "INTERNAL MODEL RESULT"}}))
    sync(**inputs, output=tmp_path / "candidate")
    en = (tmp_path / "candidate/task-docs/README.md").read_text()
    assert "INTERNAL MODEL RESULT" not in en
    assert "Validation evidence" not in en


def test_v1_sequence_starts_with_same_thirty_tasks_and_generates_unpublished_v11(inputs, tmp_path):
    first = load_manifest(AXIS_V1_CONFIG_PATH)
    receipt = sync(**{**inputs, "previous": AXIS_V1_CONFIG_PATH}, output=tmp_path / "v11")
    assert receipt["benchmark"] == "axis_v1.1"
    assert receipt["previous_benchmark"] == "axis_v1.0"
    next_round = load_manifest(tmp_path / "v11/axis_v1.1.yaml")
    assert next_round["tasks"][:30] == first["tasks"]
    assert len(next_round["tasks"]) == 40
    assert "PREVIEW — not published" in (tmp_path / "v11/task-docs/README.md").read_text()


def test_task_docs_reject_modified_receipt(inputs, tmp_path):
    receipt = sync(**inputs, output=tmp_path / "round")
    receipt["selected_task_ids"].reverse()
    (tmp_path / "round/selection.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt does not match"):
        export_docs(
            manifest_path=tmp_path / "round/axis_v1.1.yaml",
            selector_root=inputs["selector_root"],
            output=tmp_path / "docs",
        )
    assert not (tmp_path / "docs").exists()


def test_generated_yaml_uses_original_schema_and_pins_definitions(inputs, tmp_path):
    output = tmp_path / "round"
    receipt = sync(**inputs, output=output)
    original = yaml.safe_load(AXIS_V1_CONFIG_PATH.read_text())
    config = yaml.safe_load((output / "axis_v1.1.yaml").read_text())
    assert set(config) == set(original)
    assert config["tasks"][:30] == original["tasks"]
    source = load_manifest(output / config["source_manifest"])
    assert canonical_json_sha256(source) == config["source_manifest_sha256"]
    assert [task["task_id"] for task in config["tasks"]] == receipt["selected_task_ids"]
    assert all(set(task) == {"task_id", "name_zh", "instruction"} for task in config["tasks"])


def test_json_definition_input_remains_usable_and_checks_its_yaml(inputs, tmp_path):
    first = tmp_path / "first"
    sync(**inputs, output=first)
    inputs["previous"] = first / "axis_v1.1.json"
    receipt = sync(**inputs, output=tmp_path / "second")
    assert len(receipt["selected_task_ids"]) == 50
    config_path = first / "axis_v1.1.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["tasks"] = config["tasks"][:-1]
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        sync(**inputs, output=tmp_path / "tampered")
