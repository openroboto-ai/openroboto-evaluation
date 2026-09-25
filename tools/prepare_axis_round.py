#!/usr/bin/env python3
"""Freeze the first 30 AXIS tasks, then add 10 unseen tasks per weekly round.

This prepares local artifacts. Queue ingestion, scheduling and score submission
remain the responsibility of the worker/backend integration.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_runtime import canonical_json_sha256, load_manifest, task_specs
from axis_sampling import ALGORITHM, sample_tasks
from build_axis_task_subset import build
from verify_axis_release import verify_release

INITIAL_COUNT = 30
WEEKLY_ADDITION = 10


def draw_remaining(pool: dict, selected: list[int], seed: int) -> list[int]:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("sampling seed must be an integer in [0, 2**64)")
    remaining = copy.deepcopy(pool)
    selected_ids = set(selected)
    remaining["tasks"] = [task for task in pool["tasks"] if task["task_id"] not in selected_ids]
    count = WEEKLY_ADDITION if selected else INITIAL_COUNT
    if len(remaining["tasks"]) < count:
        raise ValueError(f"pool exhausted: need {count} unseen tasks, only {len(remaining['tasks'])} remain")
    return sample_tasks(remaining, count, seed)["selected_task_ids"]


def read_previous(directory: pathlib.Path, pool: dict, policy_seed: int) -> tuple[dict, list[int]]:
    record = json.loads((directory / "round.json").read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("previous round receipt must be a JSON object")
    expected = {
        "schema_version": 1,
        "algorithm": ALGORITHM,
        "pool_manifest_sha256": canonical_json_sha256(pool),
        "policy_seed": policy_seed,
        "initial_count": INITIAL_COUNT,
        "weekly_addition": WEEKLY_ADDITION,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("previous round pool, algorithm, policy seed or growth rule changed")
    history = record.get("rounds")
    if not isinstance(history, list) or not history:
        raise ValueError("previous round has no draw history")
    selected: list[int] = []
    for index, entry in enumerate(history, start=1):
        if not isinstance(entry, dict) or entry.get("round_index") != index:
            raise ValueError("previous round history is not sequential")
        added = draw_remaining(pool, selected, entry.get("sampling_seed"))
        if added != entry.get("added_task_ids"):
            raise ValueError(f"previous round {index} draw does not replay")
        selected = sorted(selected + added)
    if record.get("selected_task_ids") != selected:
        raise ValueError("previous round cumulative task list does not match its history")
    version = record.get("benchmark")
    if not isinstance(version, str) or re.fullmatch(r"axis_v[0-9]+(?:\.[0-9]+)?", version) is None:
        raise ValueError("previous round has an invalid benchmark version")
    manifest_path = directory / f"{version}.json"
    verification = verify_release(manifest_path, directory / f"{version}-tasks")
    if verification != record.get("verification"):
        raise ValueError("previous round manifest or task snapshots changed")
    manifest = load_manifest(manifest_path)
    if sorted(task_specs(manifest)) != selected or manifest.get("policy_seed") != policy_seed:
        raise ValueError("previous round manifest does not match its task list or policy seed")
    return record, selected


def prepare(
    source: pathlib.Path,
    output: pathlib.Path,
    *,
    version: str,
    seed: int,
    policy_seed: int,
    previous: pathlib.Path | None = None,
) -> dict:
    source, output = source.resolve(), output.resolve()
    if source.suffix != ".json":
        raise ValueError("source must be the frozen full-pool JSON with adjacent task snapshots")
    if re.fullmatch(r"axis_v[0-9]+(?:\.[0-9]+)?", version) is None:
        raise ValueError("version must be axis_vN.N")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("sampling seed must be an integer in [0, 2**64)")
    if type(policy_seed) is not int or not 0 <= policy_seed < 2**32:
        raise ValueError("policy seed must be an integer in [0, 2**32)")
    pool = load_manifest(source)
    task_specs(pool)
    if not isinstance(pool.get("protocol_revision"), str) or not pool["protocol_revision"].strip():
        raise ValueError("source pool must have an explicit protocol_revision")
    prior, selected = read_previous(previous.resolve(), pool, policy_seed) if previous else (None, [])
    if prior and version == prior["benchmark"]:
        raise ValueError("each round needs a new benchmark version")
    history = copy.deepcopy(prior["rounds"]) if prior else []
    added = draw_remaining(pool, selected, seed)
    selected = sorted(selected + added)
    history.append({"round_index": len(history) + 1, "sampling_seed": seed, "added_task_ids": added})
    # Exclusive creation prevents an accidental rerun from replacing a frozen
    # round. round.json is written last; without it the bundle is incomplete.
    output.mkdir(parents=True, exist_ok=False)
    try:
        manifest_path = output / f"{version}.json"
        manifest = build(
            source,
            selected,
            manifest_path,
            revision=f"{pool['protocol_revision']}__{version}",
            scope="cumulative-30-then-10-weekly; snapshot-integrity-only",
        )
        manifest["policy_seed"] = policy_seed
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        record = {
            "schema_version": 1,
            "algorithm": ALGORITHM,
            "pool_manifest_sha256": canonical_json_sha256(pool),
            "pool_task_count": len(pool["tasks"]),
            "initial_count": INITIAL_COUNT,
            "weekly_addition": WEEKLY_ADDITION,
            "policy_seed": policy_seed,
            "benchmark": version,
            "protocol_revision": manifest["protocol_revision"],
            "previous_round_sha256": canonical_json_sha256(prior) if prior else None,
            "rounds": history,
            "selected_task_ids": selected,
            "remaining_task_count": len(pool["tasks"]) - len(selected),
            "validation_scope": "snapshot-integrity-only; no model evaluation or queue deployment",
            "verification": verify_release(manifest_path, output / f"{version}-tasks"),
        }
        (output / "round.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        return record
    except BaseException:
        shutil.rmtree(output)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True, help="New, immutable round directory")
    parser.add_argument("--version", required=True, help="New local manifest identity, e.g. axis_v20260923.1")
    parser.add_argument("--seed", type=int, required=True, help="One shared draw seed per round")
    parser.add_argument("--policy-seed", type=int, required=True, help="Keep unchanged across weekly rounds")
    parser.add_argument("--previous", type=pathlib.Path, help="Previous frozen round directory; omit for first 30")
    args = parser.parse_args()
    try:
        record = prepare(
            args.source,
            args.output,
            version=args.version,
            seed=args.seed,
            policy_seed=args.policy_seed,
            previous=args.previous,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "benchmark": record["benchmark"],
                "tasks": len(record["selected_task_ids"]),
                "new_tasks": record["rounds"][-1]["added_task_ids"],
                "round_sha256": canonical_json_sha256(record),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
