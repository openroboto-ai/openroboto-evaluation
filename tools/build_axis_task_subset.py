#!/usr/bin/env python3
"""Freeze a task subset from an existing executable AXIS pool without changing tasks."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))
from axis_runtime import canonical_json_sha256, load_manifest, verify_task_payload  # noqa: E402


def build(source: pathlib.Path, task_ids: list[int], output: pathlib.Path, *, revision: str, scope: str) -> dict:
    source, output = source.resolve(), output.resolve()
    manifest = load_manifest(source)
    if manifest["protocol"]["randomization"]:
        raise ValueError("task subset builder requires a base-scene pool; randomized releases need variant snapshots")
    if output.suffix != ".json" or re.fullmatch(r"axis_v[0-9]+(?:\.[0-9]+)?", output.stem) is None:
        raise ValueError("output must be a versioned axis_vN.N.json manifest")
    if output.stem == manifest["name"] or not revision.strip() or not scope.strip():
        raise ValueError("subset needs a new version name and explicit protocol revision and pool scope")
    specs = {int(task["task_id"]): task for task in manifest["tasks"]}
    if not task_ids or len(set(task_ids)) != len(task_ids) or not set(task_ids) <= set(specs):
        raise ValueError("task ids must be a nonempty unique subset of the source pool")
    payload_root = source.parent / manifest.get("task_snapshot_root", f"{source.stem}-tasks")
    output_root = output.with_name(f"{output.stem}-tasks")
    if output.exists() or output_root.exists():
        raise FileExistsError(f"subset output already exists: {output}")
    # Validate every source before writing anything; never reconstruct a missing checker from task text.
    payloads = {
        task_id: verify_task_payload(json.loads((payload_root / f"{task_id}.json").read_text()), specs[task_id])
        for task_id in task_ids
    }
    result = copy.deepcopy(manifest)
    result.update(
        name=output.stem,
        task_snapshot_root=output_root.name,
        protocol_revision=revision,
        description="Controlled task subset copied from a frozen AXIS pool; see subset_provenance.",
        tasks=[specs[task_id] for task_id in sorted(task_ids)],
        subset_provenance={
            "pool_scope": scope,
            "source_benchmark": manifest["name"],
            "source_manifest_sha256": canonical_json_sha256(manifest),
            "source_task_ids": sorted(specs),
            "selected_task_ids": sorted(task_ids),
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    for task_id, payload in payloads.items():
        (output_root / f"{task_id}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--protocol-revision", required=True)
    parser.add_argument("--pool-scope", required=True)
    args = parser.parse_args()
    result = build(
        args.source,
        [int(value) for value in args.task_ids.split(",")],
        args.output,
        revision=args.protocol_revision,
        scope=args.pool_scope,
    )
    print(json.dumps({"manifest": str(args.output), "manifest_sha256": canonical_json_sha256(result)}))


if __name__ == "__main__":
    main()
