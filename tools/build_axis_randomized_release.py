#!/usr/bin/env python3
"""Freeze explicit upstream task bindings as a separately versioned AXIS release.

All instances must pass a real reset/render before publishing the manifest.
No seed retries, range clamps or exclusion of failing tasks are performed.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import hashlib
import json
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_randomization import ALGORITHM, PERMUTATION_ALGORITHM, load_randomization_plan, _load_strict_json
from axis_runtime import (
    AssetCache,
    AxisEnvironment,
    canonical_json_sha256,
    load_manifest,
    task_specs,
    verify_task_payload,
    DEFAULT_CACHE,
    task_runtime,
)


def _validate_instance(job: tuple) -> dict:
    from axis_perturbations import install_randomization_assets, has_domain_randomization

    relative, payload, runtime, cache_root, staging, enabled = job
    asset_cache = AssetCache(pathlib.Path(cache_root) / "assets", runtime["asset_base_url"])
    config = payload.get("official_randomization")
    if config is not None:
        install_randomization_assets(config, asset_cache.root)
    scene, _ = asset_cache.prepare_scene(payload["id"], payload["mjcf_xml"], scene_key=canonical_json_sha256(payload))
    env = AxisEnvironment(scene, payload, runtime)
    try:
        if (
            not enabled
            and config is not None
            and (config["visual"] is not None or has_domain_randomization(env.model, config["domain_randomization"]))
        ):
            raise ValueError(f"task marked fixed still has an enabled randomization distribution: {relative}")
        env.reset()
        # Fixed tasks have no AxisRandomizer, so its reset checks do not run.
        # They must meet the same validity bar before receiving a one-trial slot.
        if (
            not env.np.isfinite(env.data.qpos).all()
            or not env.np.isfinite(env.data.qvel).all()
            or env.np.any(env.data.warning.number)
        ):
            raise ValueError(f"release reset is non-finite or produced a MuJoCo warning: {relative}")
        passed, detail = env.success()
        if passed:
            raise ValueError(f"release reset already satisfies the checker: {relative}: {detail}")
        state = env.data.qpos.copy()
        image = env.render()
        wrist_camera = runtime.get("wrist_camera")
        wrist = env.render(camera=wrist_camera) if wrist_camera is not None else None
        env.reset()
        if not env.np.array_equal(state, env.data.qpos) or not env.np.array_equal(image, env.render()):
            raise ValueError(f"non-reproducible reset: {relative}")
        if wrist is not None and not env.np.array_equal(wrist, env.render(camera=wrist_camera)):
            raise ValueError(f"non-reproducible wrist image: {relative}")
        evidence = {
            "payload": relative,
            "reset_qpos_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
            "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "image_shape": list(image.shape),
            "wrist_image_sha256": hashlib.sha256(wrist.tobytes()).hexdigest() if wrist is not None else None,
            "initial_success": False,
            "official_randomization": env.reset_randomization,
        }
        if relative.endswith("official-00.json"):
            from PIL import Image

            Image.fromarray(image).save(pathlib.Path(staging) / f"preview-{payload['id']}.png")
            if wrist is not None:
                Image.fromarray(wrist).save(pathlib.Path(staging) / f"preview-{payload['id']}-wrist.png")
    finally:
        env.close()
    destination = pathlib.Path(staging) / relative
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"validated {relative}", flush=True)
    return evidence


def _validate_task(jobs: list[tuple]) -> list[dict]:
    evidence = []
    for job in jobs:
        try:
            evidence.append(_validate_instance(job))
        except Exception as error:
            # Log immediately: another task can still be running when ordered
            # pool results reach this failure. Never silently publish a subset.
            print(f"validation failed {job[0]}: {error}", flush=True)
            raise
    return evidence


def build(args: argparse.Namespace) -> dict:
    from axis_perturbations import RESET_REVISION

    source_path = pathlib.Path(args.source_manifest).resolve()
    source = load_manifest(source_path)
    request = _load_strict_json(pathlib.Path(args.bindings))
    if (
        not isinstance(request, dict)
        or set(request) != {"schema_version", "name", "protocol_revision", "namespace", "tasks", "instances"}
        or request["schema_version"] not in (1, 2)
    ):
        raise ValueError("bindings require schema_version=1, name, protocol_revision, namespace, tasks and instances")
    if request["name"] == source["name"] or request["protocol_revision"] == source.get("protocol_revision"):
        raise ValueError("randomization requires a new benchmark name and protocol revision")
    output = pathlib.Path(args.output).resolve()
    if output.exists():
        raise ValueError(f"output already exists; use a new release directory: {output}")
    specs = task_specs(source)
    bindings = request["tasks"]
    if not isinstance(bindings, list):
        raise ValueError("bindings.tasks must be a list")
    by_id = {}
    mixed = request["schema_version"] == 2
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {
            "task_id",
            "source_payload_sha256",
            "upstream_provenance",
            "domain_randomization",
            "object_order",
            "visual",
        } | ({"randomization_enabled", "runtime_overrides"} if mixed else set()):
            raise ValueError(
                "each task binding requires id, payload hash, provenance, DR config/order and visual binding"
            )
        tid = binding["task_id"]
        if type(tid) is not int or tid in by_id or tid not in specs:
            raise ValueError(f"duplicate or unknown task binding: {tid}")
        if not isinstance(binding["upstream_provenance"], dict) or not binding["upstream_provenance"]:
            raise ValueError(f"task {tid} requires upstream provenance")
        by_id[tid] = binding
        if mixed and type(binding["randomization_enabled"]) is not bool:
            raise ValueError("randomization_enabled must be a boolean")
        if mixed and not binding["randomization_enabled"] and binding["visual"] is not None:
            raise ValueError("fixed tasks must disable all randomization")
    if set(by_id) != set(specs):
        raise ValueError("bindings must cover the entire selected source manifest; no silent task exclusions")
    instances = request["instances"]
    if not isinstance(instances, list) or len(instances) < 2:
        raise ValueError("at least two explicit instances are required")
    seen = set()
    for instance in instances:
        if not isinstance(instance, dict) or set(instance) != {
            "variant_id",
            "submit_nonce",
            "global_seed",
            "attempt_id",
            "render_variant_id",
            "replica_id",
        }:
            raise ValueError("instance fields must freeze variant_id, submit_nonce and all upstream render seed inputs")
        if not isinstance(instance["variant_id"], str) or instance["variant_id"] in seen:
            raise ValueError("instance variant_id must be a unique string")
        seen.add(instance["variant_id"])
    benchmark = copy.deepcopy(source)
    benchmark.update(
        name=request["name"],
        protocol_revision=request["protocol_revision"],
        task_snapshot_root="payloads",
        description=f"Frozen AXIS reset/render instances derived from {source['name']}.",
    )
    benchmark["protocol"].update(
        randomization=True, scene_policy="frozen-official-instances", randomization_seed_default=0
    )
    benchmark["protocol"]["randomization_manifest"] = "randomization.json"
    benchmark["runtime"]["scene_policy"] = "frozen-official-instances"
    if mixed:
        benchmark["protocol"]["score_reduction"] = "task_mean"
        for spec in benchmark["tasks"]:
            binding = by_id[spec["task_id"]]
            spec.update(
                randomization_enabled=binding["randomization_enabled"], runtime_overrides=binding["runtime_overrides"]
            )
            task_runtime(benchmark, spec)
    elif any(binding["visual"] is not None for binding in bindings):
        benchmark["runtime"].update(image_width=640, image_height=360)
        wrists = {
            "wrist" if binding["visual"].get("mode") == "official_franka_v6" else binding["visual"]["wrist_camera"]
            for binding in bindings
            if binding["visual"] is not None
        }
        if len(wrists) != 1 or any(binding["visual"] is None for binding in bindings):
            raise ValueError("all tasks in a visual release must share the same declared camera contract")
        benchmark["runtime"]["wrist_camera"] = wrists.pop()
        fronts = {
            "frontview" if binding["visual"].get("mode") == "official_franka_v6" else binding["visual"]["front_camera"]
            for binding in bindings
        }
        if len(fronts) != 1:
            raise ValueError("all tasks must share the same front camera contract")
        benchmark["runtime"]["camera"] = fronts.pop()
    plan = {
        "schema_version": 2 if mixed else 1,
        "benchmark": request["name"],
        "protocol_revision": request["protocol_revision"],
        "seed_contract": {
            "source": "queue",
            "algorithm": PERMUTATION_ALGORITHM if mixed else ALGORITHM,
            "namespace": request["namespace"],
        },
        "tasks": [],
    }
    source_tasks = source_path.parent / source.get("task_snapshot_root", source_path.stem + "-tasks")
    payloads = {}
    for tid, spec in specs.items():
        base = verify_task_payload(_load_strict_json(source_tasks / f"{tid}.json"), spec)
        binding = by_id[tid]
        if canonical_json_sha256(base) != binding["source_payload_sha256"]:
            raise ValueError(f"task {tid} binding does not match its frozen source payload")
        variants = []
        enabled = binding.get("randomization_enabled", True)
        for instance in instances if enabled else instances[:1]:
            payload = copy.deepcopy(base)
            visual = copy.deepcopy(binding["visual"])
            if visual is not None:
                visual.update(
                    global_seed=instance["global_seed"],
                    attempt_id=instance["attempt_id"],
                    variant_id=instance["render_variant_id"],
                    replica_id=instance["replica_id"],
                )
                front_camera = (
                    "frontview"
                    if visual.get("mode")
                    in {"official_franka_v6", "official_franka_components", "official_franka_components_v2"}
                    else visual["front_camera"]
                )
                if not mixed and front_camera != benchmark["runtime"]["camera"]:
                    raise ValueError("front_camera must match the policy observation camera")
            config = {
                "schema_version": 1,
                "upstream_reset_revision": RESET_REVISION,
                "submit_nonce": instance["submit_nonce"],
                "domain_randomization": binding["domain_randomization"],
                "object_order": binding["object_order"],
                "visual": visual,
            }
            if enabled or config["domain_randomization"] != {}:
                payload["official_randomization"] = config
            variant = {
                "variant_id": instance["variant_id"],
                "payload_path": f"payloads/{tid}-{instance['variant_id']}.json",
                "payload_canonical_sha256": canonical_json_sha256(payload),
                "mjcf_sha256": spec["mjcf_sha256"],
                "checker_sha256": spec["checker_sha256"],
                "initial_state_sha256": spec["initial_state_sha256"],
                "dimensions": {
                    "official-contract": config,
                    "upstream-provenance": binding["upstream_provenance"],
                    "source-payload-sha256": binding["source_payload_sha256"],
                },
            }
            if "official_randomization" in payload:
                variant["official_randomization_sha256"] = canonical_json_sha256(config)
            variants.append(variant)
            payloads[variant["payload_path"]] = payload
        plan["tasks"].append({
            "task_id": tid,
            "instruction": spec["instruction"],
            "variants": variants,
            **({"randomization_enabled": enabled} if mixed else {}),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".axis-official-", dir=output.parent) as temporary:
        staging = pathlib.Path(temporary)
        (staging / "randomization.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        (staging / "benchmark.json").write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n")
        load_manifest(staging / "benchmark.json")
        load_randomization_plan(
            staging / "randomization.json",
            expected_benchmark=request["name"],
            expected_protocol_revision=request["protocol_revision"],
            benchmark_task_specs=task_specs(benchmark),
        )
        jobs = [
            (
                relative,
                payload,
                task_runtime(benchmark, task_specs(benchmark)[payload["id"]]),
                str(pathlib.Path(args.cache_root).resolve()),
                str(staging),
                by_id[payload["id"]].get("randomization_enabled", True),
            )
            for relative, payload in payloads.items()
        ]
        workers = getattr(args, "workers", 1)
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if workers == 1:
            evidence = [_validate_instance(job) for job in jobs]
        else:
            # Keep each task's native GL resources in a fresh process. Reusing
            # a renderer process across unrelated MJCFs can crash the driver.
            grouped = {}
            for job in jobs:
                grouped.setdefault(job[1]["id"], []).append(job)
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn"), max_tasks_per_child=1
            ) as pool:
                evidence = [row for rows in pool.map(_validate_task, grouped.values()) for row in rows]
        report = {
            "schema_version": 1,
            "benchmark": request["name"],
            "instances": evidence,
            "source_manifest_sha256": canonical_json_sha256(source),
            "bindings_sha256": canonical_json_sha256(request),
            "validation": "reset, render and exact repeat; no policy rollout or task solvability claim",
        }
        (staging / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (staging / "bindings.json").write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
        staging.rename(output)
    return {"output": str(output), "tasks": len(specs), "instances": len(evidence)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    print(json.dumps(build(args), indent=2))
