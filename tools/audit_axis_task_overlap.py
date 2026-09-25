#!/usr/bin/env python3
"""Conservatively detect success predicates already implied by another frozen AXIS task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))
from axis_runtime import canonical_json_sha256, load_manifest, verify_task_payload  # noqa: E402


def conjunction(config: dict) -> list[dict] | None:
    if config.get("type") != "CompositeChecker":
        return [config]
    if str(config.get("operator", "AND")).upper() != "AND":
        return None
    leaves = []
    for child in config.get("checkers") or []:
        nested = conjunction(child)
        if nested is None:
            return None
        leaves.extend(nested)
    return leaves or None


def leaf_implication(source: dict, target: dict) -> str | None:
    if source == target:
        return "identical-checker"
    if source.get("type") != "RelativeCylinderChecker":
        return None
    if not all(source.get(key) and source.get(key) == target.get(key) for key in ("objName", "refName")):
        return None
    radius = float(source.get("xyRadius", 0.06))
    lower, upper = float(source.get("heightMin", 0.0)), float(source.get("heightMax", 0.03))
    if not all(math.isfinite(v) for v in (radius, lower, upper)) or radius <= 0 or lower >= upper:
        return None
    if target.get("type") == "RelativeCylinderChecker":
        other = (
            float(target.get("xyRadius", 0.06)),
            float(target.get("heightMin", 0.0)),
            float(target.get("heightMax", 0.03)),
        )
        if all(math.isfinite(v) for v in other) and radius <= other[0] and lower >= other[1] and upper <= other[2]:
            return "narrower-cylinder"
    if target.get("type") == "RelativePositionBoundsChecker":
        intervals = ((-radius, radius), (-radius, radius), (lower, upper))
        constrained = False
        for key, (minimum, maximum) in zip(("xRange", "yRange", "zRange"), intervals):
            bounds = target.get(key)
            if bounds is None:
                continue
            constrained = True
            if len(bounds) != 2 or not all(math.isfinite(float(v)) for v in bounds):
                return None
            if float(bounds[0]) > minimum or float(bounds[1]) < maximum:
                return None
        if constrained:
            return "cylinder-enclosed-by-position-box"
    return None


def predicate_implication(source: dict, target: dict) -> list[dict] | None:
    left, right = conjunction(source), conjunction(target)
    if left is None or right is None:
        return None
    proof = []
    for target_index, target_leaf in enumerate(right):
        for source_index, source_leaf in enumerate(left):
            rule = leaf_implication(source_leaf, target_leaf)
            if rule:
                proof.append({"source_conjunct": source_index, "target_conjunct": target_index, "rule": rule})
                break
        else:
            return None
    return proof


def audit(manifest_path: pathlib.Path) -> dict:
    manifest = load_manifest(manifest_path)
    if manifest["protocol"]["randomization"]:
        raise ValueError("predicate overlap audit requires a frozen base-scene pool")
    payload_dir = manifest_path.with_name(f"{manifest_path.stem}-tasks")
    specs = {int(task["task_id"]): task for task in manifest["tasks"]}
    roots = {}
    for task_id, spec in specs.items():
        payload = verify_task_payload(json.loads((payload_dir / f"{task_id}.json").read_text()), spec)
        roots[task_id] = payload["checker_config"].get("checker", payload["checker_config"])
    implications = []
    for source_id, source in sorted(specs.items()):
        for target_id, target in sorted(specs.items()):
            if source_id == target_id or any(
                source[key] != target[key] for key in ("mjcf_sha256", "initial_state_sha256")
            ):
                continue
            proof = predicate_implication(roots[source_id], roots[target_id])
            if proof is not None:
                implications.append({"source_task_id": source_id, "target_task_id": target_id, "proof": proof})
    return {
        "schema_version": 1,
        "manifest_sha256": canonical_json_sha256(manifest),
        "runtime_source_sha256": hashlib.sha256((ROOT / "libero_eval/axis_runtime.py").read_bytes()).hexdigest(),
        "meaning": "Every state passing the source predicate also passes the target under the current frozen runtime.",
        "limitations": (
            "Conservative AND/leaf rules on identical scene and initial state; "
            "no implication found does not prove independence. This is not a causal analysis of policy rollouts."
        ),
        "implications": implications,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = audit(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"implications": report["implications"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
