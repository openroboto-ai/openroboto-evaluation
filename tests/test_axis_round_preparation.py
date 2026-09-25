"""Cumulative weekly draws must preserve old tasks and their frozen definitions."""

import copy
import json
import pathlib
import shutil
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_backend import apply_manifest_defaults
from axis_runtime import canonical_json_sha256, load_manifest
from prepare_axis_round import prepare, read_previous
from verify_axis_release import verify_release


@pytest.fixture
def pool(tmp_path):
    original = load_manifest(ROOT / "configs/benchmarks/axis_v1.0.json")
    original["protocol_revision"] = "test-pool-v1"
    payload = json.loads((ROOT / "configs/benchmarks/axis_v1.0-tasks/501.json").read_text())
    template = next(task for task in original["tasks"] if task["task_id"] == 501)
    original["tasks"] = [dict(template, task_id=task_id) for task_id in range(1, 56)]
    source = tmp_path / "axis_v1.0.json"
    source.write_text(json.dumps(original))
    snapshots = tmp_path / "axis_v1.0-tasks"
    snapshots.mkdir()
    for spec in original["tasks"]:
        (snapshots / f"{spec['task_id']}.json").write_text(json.dumps(dict(payload, id=spec["task_id"])))
    return source


def freeze(pool, output, index=1, previous=None, seed=23):
    return prepare(pool, output, version=f"axis_v1.{index}", seed=seed, policy_seed=7, previous=previous)


def test_three_rounds_are_30_40_50_and_keep_all_old_definitions(pool, tmp_path):
    previous, selected = None, set()
    for index, expected in enumerate((30, 40, 50), start=1):
        directory = tmp_path / f"round{index}"
        record = freeze(pool, directory, index, previous, seed=index)
        current = set(record["selected_task_ids"])
        assert len(current) == expected
        assert selected <= current
        assert set(record["rounds"][-1]["added_task_ids"]) == current - selected
        manifest = load_manifest(directory / f"axis_v1.{index}.json")
        source_specs = {spec["task_id"]: spec for spec in load_manifest(pool)["tasks"]}
        assert all(spec == source_specs[spec["task_id"]] for spec in manifest["tasks"])
        if previous:
            prior = json.loads((previous / "round.json").read_text())
            assert record["previous_round_sha256"] == canonical_json_sha256(prior)
        read_previous(directory, load_manifest(pool), 7)
        previous, selected = directory, current
    with pytest.raises(ValueError, match="pool exhausted"):
        freeze(pool, tmp_path / "round4", 4, previous)
    assert not (tmp_path / "round4").exists()


def test_round_is_reproducible_and_relocatable_without_source_payload_directory(pool, tmp_path):
    first = freeze(pool, tmp_path / "a")
    replay = freeze(pool, tmp_path / "b")
    assert first == replay
    changed_seed = freeze(pool, tmp_path / "c", seed=24)
    assert first["selected_task_ids"] != changed_seed["selected_task_ids"]
    shutil.rmtree(pool.with_name("axis_v1.0-tasks"))
    relocated = tmp_path / "relocated"
    shutil.move(tmp_path / "a", relocated)
    args = SimpleNamespace(
        benchmark="axis",
        axis_manifest=str(relocated / "axis_v1.1.json"),
        axis_replan_steps=None,
        seed=None,
        config=None,
    )
    apply_manifest_defaults(args)
    assert args.seed == 7
    assert len(args.axis_loaded_manifest["tasks"]) == 30
    assert verify_release(relocated / "axis_v1.1.json", relocated / "axis_v1.1-tasks") == first["verification"]


@pytest.mark.parametrize("mutation", ["pool", "policy_seed", "draw", "missing_seed", "task_list", "payload"])
def test_changed_previous_round_is_rejected_before_output(pool, tmp_path, mutation):
    prior_dir = tmp_path / "prior"
    record = freeze(pool, prior_dir)
    policy_seed = 7
    if mutation == "pool":
        manifest = load_manifest(pool)
        manifest["description"] = "changed pool"
        pool.write_text(json.dumps(manifest))
    elif mutation == "policy_seed":
        policy_seed = 8
    elif mutation == "payload":
        task_id = record["selected_task_ids"][0]
        path = prior_dir / "axis_v1.1-tasks" / f"{task_id}.json"
        payload = json.loads(path.read_text())
        payload["mjcf_xml"] += "<!-- drift -->"
        path.write_text(json.dumps(payload))
    else:
        record = copy.deepcopy(record)
        if mutation == "draw":
            record["rounds"][0]["added_task_ids"].reverse()
        elif mutation == "missing_seed":
            del record["rounds"][0]["sampling_seed"]
        else:
            record["selected_task_ids"].pop()
        (prior_dir / "round.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        prepare(pool, tmp_path / "next", version="axis_v1.2", seed=24, policy_seed=policy_seed, previous=prior_dir)
    assert not (tmp_path / "next").exists()


def test_rerun_does_not_replace_a_round_and_failure_leaves_no_partial_bundle(pool, tmp_path):
    output = tmp_path / "round"
    record = freeze(pool, output)
    with pytest.raises(FileExistsError):
        freeze(pool, output, seed=24)
    assert json.loads((output / "round.json").read_text()) == record
    selected_id = record["selected_task_ids"][0]
    (pool.with_name("axis_v1.0-tasks") / f"{selected_id}.json").unlink()
    with pytest.raises(FileNotFoundError):
        freeze(pool, tmp_path / "broken")
    assert not (tmp_path / "broken").exists()


def test_release_verifier_accepts_mixed_length_ids_and_rejects_extra_snapshots(pool):
    snapshots = pool.with_name("axis_v1.0-tasks")
    assert len(verify_release(pool, snapshots)["tasks"]) == 55
    (snapshots / "999.json").write_text("{}")
    with pytest.raises(ValueError, match="snapshot set mismatch"):
        verify_release(pool, snapshots)
