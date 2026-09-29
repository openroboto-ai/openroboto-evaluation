#!/usr/bin/env python3
"""Audit all planned physical reset seeds without policy inference or visual changes.

This does not publish a release or replace failing seeds. It records whether a
task can retain its proposed physical reset, must keep its base object poses,
or needs an upstream task-definition correction before release validation.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))
from axis_perturbations import AxisRandomizer, RESET_REVISION
from axis_runtime import (
    AssetCache,
    AxisEnvironment,
    DEFAULT_CACHE,
    canonical_json_sha256,
    load_manifest,
    task_specs,
    verify_task_payload,
)


def audit_task(job):
    spec, payload, binding, instances, runtime, cache_root, reports = job
    cache = AssetCache(cache_root / "assets", runtime["asset_base_url"])
    scene, _ = cache.prepare_scene(spec["task_id"], payload["mjcf_xml"], scene_key=canonical_json_sha256(payload))
    config = dict(
        schema_version=1,
        upstream_reset_revision=RESET_REVISION,
        submit_nonce=instances[0]["submit_nonce"],
        domain_randomization={},
        object_order=[],
        visual=None,
    )
    physical_payload = {**payload, "official_randomization": config}
    env = AxisEnvironment(scene, physical_payload, {**runtime, "image_width": 16, "image_height": 9})

    def probe(cfg):
        env.axis_randomizer = AxisRandomizer(
            env.model, cfg, asset_root=cache.root, task_id=payload["id"], task_name=payload["name"], width=16, height=9
        )
        try:
            env.reset()
            return dict(valid=True, qpos_sha256=hashlib.sha256(env.data.qpos.tobytes()).hexdigest())
        except ValueError as error:
            return dict(valid=False, error=str(error))

    try:
        base = probe(config)
        active = binding["upstream_provenance"]["enabled_components"]["physical_reset"]
        trials = {}
        for instance in instances if active else instances[:1]:
            cfg = {
                **config,
                "domain_randomization": binding["domain_randomization"],
                "object_order": binding["object_order"],
                "submit_nonce": instance["submit_nonce"],
            }
            trials[instance["variant_id"]] = probe(cfg)
        valid = all(t["valid"] for t in trials.values())
        recommendation = "retain" if valid else "disable_physical_reset" if base["valid"] else "blocked"
        result = dict(
            task_id=spec["task_id"],
            source_payload_sha256=canonical_json_sha256(payload),
            physical_reset_enabled=active,
            base=base,
            trials=trials,
            recommendation=recommendation,
        )
        (reports / f"{spec['task_id']}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    task_id=spec["task_id"],
                    recommendation=recommendation,
                    valid=sum(t["valid"] for t in trials.values()),
                    trials=len(trials),
                )
            ),
            flush=True,
        )
        return result
    finally:
        env.close()


def audit(args):
    source = load_manifest(args.source_manifest)
    specs = task_specs(source)
    bindings = json.loads(args.bindings.read_bytes())
    snapshots = args.source_manifest.parent / source.get("task_snapshot_root", args.source_manifest.stem + "-tasks")
    reports = args.output.with_name(args.output.stem + "-tasks")
    reports.mkdir(parents=True, exist_ok=True)
    jobs = []
    if {b["task_id"] for b in bindings["tasks"]} != set(specs):
        raise ValueError("reset audit bindings must cover the whole source manifest")
    for binding in bindings["tasks"]:
        tid = binding["task_id"]
        payload = verify_task_payload(json.loads((snapshots / f"{tid}.json").read_bytes()), specs[tid])
        if canonical_json_sha256(payload) != binding["source_payload_sha256"]:
            raise ValueError(f"task {tid} binding source hash mismatch")
        jobs.append((specs[tid], payload, binding, bindings["instances"], source["runtime"], args.cache_root, reports))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"), max_tasks_per_child=1) as pool:
        rows = list(pool.map(audit_task, jobs))
    result = dict(
        schema_version=1,
        scope="physical reset validity only; no policy scores or full visual instance validation",
        source_manifest_sha256=canonical_json_sha256(source),
        bindings_sha256=canonical_json_sha256(bindings),
        task_count=len(rows),
        tasks=rows,
        recommendations={
            kind: sum(row["recommendation"] == kind for row in rows)
            for kind in ["retain", "disable_physical_reset", "blocked"]
        },
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if getattr(args, "validated_bindings", None):
        args.validated_bindings.write_text(json.dumps(validated_bindings(bindings, result), indent=2) + "\n")
    return {k: v for k, v in result.items() if k != "tasks"}


def validated_bindings(bindings, report):
    """Disable invalid physical components without changing seeds or other parts."""
    if canonical_json_sha256(bindings) != report["bindings_sha256"]:
        raise ValueError("reset audit belongs to different bindings")
    rows = {row["task_id"]: row for row in report["tasks"]}
    if len(rows) != len(report["tasks"]) or set(rows) != {b["task_id"] for b in bindings["tasks"]}:
        raise ValueError("reset audit must cover all bindings exactly once")
    result = copy.deepcopy(bindings)
    audit_hash = canonical_json_sha256(report)
    for binding in result["tasks"]:
        row = rows[binding["task_id"]]
        if row["source_payload_sha256"] != binding["source_payload_sha256"]:
            raise ValueError("reset audit source payload mismatch")
        expected = {i["variant_id"] for i in bindings["instances"]}
        if not row["physical_reset_enabled"]:
            expected = {bindings["instances"][0]["variant_id"]}
        if set(row["trials"]) != expected:
            raise ValueError("reset audit does not cover all required frozen seeds")
        if all(trial["valid"] for trial in row["trials"].values()):
            continue
        if not row["base"]["valid"]:
            raise ValueError(f"task {binding['task_id']} has no valid audited base reset; cannot release")
        binding.update(domain_randomization={}, object_order=[])
        provenance = binding["upstream_provenance"]
        provenance["enabled_components"]["physical_reset"] = False
        provenance["unavailable_components"]["physical_reset"] = (
            "physical reset produces invalid frozen instances; keep the valid source pose"
        )
        provenance["physical_reset_audit"] = {
            "report_sha256": audit_hash,
            "failed_variants": [key for key, trial in row["trials"].items() if not trial["valid"]],
        }
        binding["randomization_enabled"] = binding["visual"] is not None
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validated-bindings", type=Path, help="Write bindings with invalid physical resets disabled")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    print(json.dumps(audit(args), indent=2))
