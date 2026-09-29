#!/usr/bin/env python3
"""Select independently supported, pinned AXIS components per task.

This is an offline capability check, never a model-dependent fallback. A release
still needs build_axis_randomized_release.py to validate every frozen instance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))
from axis_scene import CAMERA_CONFIG_SHA256, prepare_scene, resolve_profile
from axis_perturbations import (
    RESET_REVISION,
    AxisRandomizer,
    install_randomization_assets,
    has_domain_randomization,
)
from axis_runtime import (
    AssetCache,
    DEFAULT_CACHE,
    JOINT_NAMES,
    canonical_json_sha256,
    load_manifest,
    verify_task_payload,
)


def visual_config(components: dict) -> dict:
    return dict(
        mode="official_franka_components_v2",
        camera_config_sha256=CAMERA_CONFIG_SHA256,
        global_seed=42,
        attempt_id=0,
        variant_id=0,
        replica_id=0,
        components=components,
    )


def native_surfaces(model) -> dict:
    """Only unambiguously named world planes are native floor candidates.

    Existing material/texture sharing is checked by AxisRandomizer. Mesh
    names alone are insufficient evidence that an object is a table surface.
    """
    names = [
        model.geom(i).name
        for i in range(model.ngeom)
        if int(model.geom_type[i]) == 0 and model.geom_bodyid[i] == 0 and model.geom(i).name
    ]
    return {"floor": names} if names else {}


def frozen_instances(namespace: str) -> list[dict]:
    """Spread seed inputs across the nonce/recipe domains before evaluation.

    Adjacent integer nonces are correlated in the backend's first xorshift draw.
    Hash-derived inputs avoid that correlation without altering upstream ranges
    or sampling a second seed in response to validation/model outcomes.
    """

    def rank(kind, index):
        return int.from_bytes(hashlib.sha256(f"{namespace}:{kind}:{index}".encode()).digest(), "big")

    recipes = sorted(range(8640), key=lambda index: rank("recipe", index))[:20]
    return [
        dict(
            variant_id=f"official-{i:02d}",
            submit_nonce=f"{rank('nonce', i) % 100_000_000:08d}",
            global_seed=42,
            attempt_id=0,
            render_variant_id=recipes[i],
            replica_id=i % 4,
        )
        for i in range(20)
    ]


def arena_geometry_check(path: Path, camera_config: dict, model, components: dict) -> None:
    """Reject an upstream replacement table that would move task geometry apart."""
    import mujoco
    import numpy as np

    if not np.isclose(model.body("franka/").pos[2], 0, atol=1e-6):
        raise ValueError("source robot base is elevated; preserve its original supporting geometry")
    xml, metadata = prepare_scene(path, camera_config, components)
    render_model = mujoco.MjModel.from_xml_string(xml)
    physical, rendered = mujoco.MjData(model), mujoco.MjData(render_model)
    if model.nq != render_model.nq:
        raise ValueError("arena changes the state dimension")
    rendered.qpos[:] = physical.qpos
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            rendered.qpos[model.jnt_qposadr[j] + 2] += metadata["free_joint_qpos_z_offset"]
    mujoco.mj_forward(model, physical)
    mujoco.mj_forward(render_model, rendered)
    offset = np.array([0, 0, metadata["arm_scene_z_offset"]])
    for i in range(1, model.nbody):
        name = model.body(i).name
        if not name:
            # Named geoms/joints identify all task bodies in published AXIS;
            # an anonymous frame cannot safely be matched after shell removal.
            raise ValueError("unnamed task body cannot certify uniform scene translation")
        try:
            ri = render_model.body(name).id
        except KeyError:
            # Only the source table shell may disappear in the upstream merge.
            if "table" not in name.lower() or model.body_jntnum[i]:
                raise ValueError(f"arena removed task body {name!r}")
            continue
        if not np.allclose(rendered.xpos[ri], physical.xpos[i] + offset, atol=1e-6):
            raise ValueError(f"arena does not translate task body {name!r} with the robot")
        if not np.allclose(rendered.xmat[ri], physical.xmat[i], atol=1e-6):
            raise ValueError(f"arena changes task body orientation {name!r}")


def _without_robot_joint_pose(initial):
    """Only the nine evaluator-controlled robot joint values may differ.

    The reset adapter cannot perturb these hinge/slide joints. Keep
    all object poses, robot base poses, mocap state, and unknown fields intact.
    """
    result = copy.deepcopy(initial) if initial is not None else {}
    if not isinstance(result, dict):
        return result
    robots = result.get("robots")
    if isinstance(robots, dict) and isinstance(robots.get("franka"), dict):
        robot = robots["franka"]
        dofs = robot.get("dof_pos")
        if isinstance(dofs, dict):
            for name in JOINT_NAMES:
                dofs.pop(name.split("/", 1)[1], None)
            if not dofs:
                robot.pop("dof_pos")
        if not robot:
            robots.pop("franka")
        if not robots:
            result.pop("robots")
    return result


def matching_reset_source(payload: dict, source: dict | None) -> bool:
    if source is None or "domain_randomization" not in source:
        return False
    if any(source.get(key) != payload[key] for key in ("id", "mjcf_xml", "checker_config")):
        return False
    return _without_robot_joint_pose(source.get("initial_state")) == _without_robot_joint_pose(payload["initial_state"])


def plan_task(payload: dict, spec: dict, scene: Path, asset_root: Path, source_record: dict | None) -> dict:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    components = dict(arena=False, front_camera=False, wrist_camera=False, background=False, surfaces={})
    reasons = {}
    visual = visual_config(components)
    camera_config, profile = resolve_profile(visual, payload["id"], payload["name"])
    domain = {}
    provenance = {"camera_config_sha256": CAMERA_CONFIG_SHA256, "reset_revision": RESET_REVISION}
    if matching_reset_source(payload, source_record):
        domain = source_record["domain_randomization"]
        provenance["reset_source_record_sha256"] = canonical_json_sha256(source_record)
        provenance["reset_source"] = "matching API task snapshot; null retains the backend default"
        provenance["robot_joint_initialization_differs"] = (
            source_record.get("initial_state") != payload["initial_state"]
        )
    else:
        reasons["physical_reset"] = "no reset configuration with matching task ID/MJCF/checker/non-robot initial state"
    reset_config = dict(
        schema_version=1,
        upstream_reset_revision=RESET_REVISION,
        submit_nonce="00123456",
        domain_randomization=domain,
        object_order=list((domain or {}).get("objects", {})),
        visual=None,
    )
    # A malformed configuration is an error, never a quiet downgrade.
    AxisRandomizer(
        model,
        reset_config,
        asset_root=asset_root,
        task_id=payload["id"],
        task_name=payload["name"],
        width=320,
        height=180,
    )
    for component in ("front_camera", "wrist_camera", "background"):
        candidate = {**components, component: True}
        try:
            prepare_scene(scene, camera_config, candidate, source_frame=True)
            components[component] = True
        except ValueError as exc:
            reasons[component] = str(exc)
    arena = {**components, "arena": True, "background": True, "surfaces": {"table": [], "floor": [], "wall": []}}
    try:
        arena_geometry_check(scene, camera_config, model, arena)
        components = arena
    except ValueError as exc:
        reasons["arena"] = str(exc)
        for surface, names in native_surfaces(model).items():
            candidate = {**components, "surfaces": {surface: names}}
            candidate_config = {
                **reset_config,
                "domain_randomization": {},
                "object_order": [],
                "visual": visual_config(candidate),
            }
            install_randomization_assets(candidate_config, asset_root)
            from axis_scene import AxisSceneRenderer

            renderer = None
            try:
                renderer = AxisSceneRenderer(
                    scene, {**payload, "official_randomization": candidate_config}, model, width=320, height=180
                )
                components["surfaces"][surface] = names
            except ValueError as error:
                reasons[surface + "_material"] = str(error)
            finally:
                if renderer is not None:
                    renderer.renderer.close()
    enabled_visual = any(components[k] for k in ("arena", "front_camera", "wrist_camera", "background", "surfaces"))
    physical_enabled = has_domain_randomization(model, domain)
    provenance.update(
        enabled_components={**components, "physical_reset": physical_enabled}, unavailable_components=reasons
    )
    runtime = (
        dict(
            camera="frontview" if components["front_camera"] else "camera0",
            wrist_camera="wrist" if components["wrist_camera"] else None,
            image_width=320,
            image_height=180,
        )
        if enabled_visual
        else {}
    )
    return dict(
        task_id=payload["id"],
        source_payload_sha256=canonical_json_sha256(payload),
        upstream_provenance=provenance,
        domain_randomization=domain,
        object_order=list((domain or {}).get("objects", {})),
        visual=visual_config(components) if enabled_visual else None,
        randomization_enabled=enabled_visual or physical_enabled,
        runtime_overrides=runtime,
    )


def _plan_spec(job):
    spec, snapshots, records, cache_root, asset_url, *reports = job
    tid = spec["task_id"]
    payload = verify_task_payload(json.loads((snapshots / f"{tid}.json").read_bytes()), spec)
    raw_path = records / f"{tid}.json" if records else None
    record = json.loads(raw_path.read_bytes()) if raw_path and raw_path.is_file() else None
    assets = AssetCache(cache_root / "assets", asset_url)
    scene, _ = assets.prepare_scene(tid, payload["mjcf_xml"], scene_key=canonical_json_sha256(payload))
    binding = plan_task(payload, spec, scene, assets.root, record)
    if reports:
        destination = reports[0] / f"{tid}.json"
        destination.write_text(json.dumps(binding, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"task_id": tid, **binding["upstream_provenance"]["enabled_components"]}), flush=True)
    return binding


def plan(args) -> dict:
    source = load_manifest(args.source_manifest)
    snapshots = args.source_manifest.parent / source.get("task_snapshot_root", args.source_manifest.stem + "-tasks")
    reports = args.output.with_name(args.output.stem + "-tasks")
    reports.mkdir(parents=True, exist_ok=True)
    jobs = [
        (spec, snapshots, args.source_records, args.cache_root, source["runtime"]["asset_base_url"], reports)
        for spec in source["tasks"]
    ]
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    # Native GL contexts and mesh compiler state must not leak between task
    # probes or be inherited through fork. Isolate serial probes as well.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"), max_tasks_per_child=1) as pool:
        bindings = list(pool.map(_plan_spec, jobs))
    return dict(
        schema_version=2,
        name=args.name,
        protocol_revision=args.name + "_official_components_native_mujoco_v2",
        namespace=args.name + "-official-components-v1",
        tasks=bindings,
        instances=frozen_instances(args.name + "-official-components-v1"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-records", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    result = plan(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
