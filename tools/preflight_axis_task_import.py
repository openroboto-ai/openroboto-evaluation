#!/usr/bin/env python3
"""Load/render imported AXIS scenes and check for success without robot motion.

This is a structural and negative-control audit, not a successful-action
qualification. Run with the pinned AXIS runtime Python and MUJOCO_GL=osmesa.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))
from axis_runtime import AssetCache, AxisEnvironment, _runtime_state, load_manifest  # noqa: E402


def inspect(path, output, steps):
    from PIL import Image

    payload = json.loads(path.read_text())
    task_id = payload["id"]
    result = {"task_id": task_id, "instruction": payload["name"], "positive_control_qualified": False}
    environment = None
    try:
        runtime = load_manifest()["runtime"]
        cache = AssetCache(ROOT / ".cache/axis/assets", runtime["asset_base_url"], workers=4)
        scene, assets = cache.prepare_scene(task_id, payload["mjcf_xml"], scene_key="legacy-import-v1")
        environment = AxisEnvironment(scene, payload, runtime)
        environment.reset()
        import numpy as np

        critical_contacts = []
        for index, contact in enumerate(environment.data.contact):
            if float(contact.dist) >= -0.002:
                continue
            geoms = [int(value) for value in contact.geom]
            plane = environment.mujoco.mjtGeom.mjGEOM_PLANE
            if not any(environment.model.geom_type[geom] == plane for geom in geoms):
                continue
            other = next(geom for geom in geoms if environment.model.geom_type[geom] != plane)
            body = int(environment.model.geom_bodyid[other])
            ancestor = body
            dynamic = False
            while ancestor:
                dynamic = dynamic or bool(environment.model.body_jntnum[ancestor])
                ancestor = int(environment.model.body_parentid[ancestor])
            if not dynamic:
                continue
            force = np.zeros(6)
            environment.mujoco.mj_contactForce(environment.model, environment.data, index, force)
            # Soft free-body contacts can briefly penetrate a few millimetres.
            # The invalid fixed-height hinge case produces singular forces.
            if float(force[0]) < 1e6:
                continue
            critical_contacts.append({
                "body": environment.mujoco.mj_id2name(environment.model, environment.mujoco.mjtObj.mjOBJ_BODY, body),
                "penetration_m": -float(contact.dist),
                "normal_force_n": float(force[0]),
            })
        result["critical_dynamic_ground_contacts"] = critical_contacts
        result["ground_clearance_ok"] = not critical_contacts
        Image.fromarray(environment.render()).save(output / f"{task_id}.png")
        result.update(scene_and_control_ok=True, assets=assets, scene=str(scene))
        state = _runtime_state(environment.mujoco, environment.model, environment.data)
        result["initial_state"] = vars(state)
        success, detail = environment.success()
        result.update(checker_implemented=True, initial_success=success, initial_checker=detail)
        action = environment.observation_state()
        first_success = 0 if success else None
        for step in range(1, steps + 1):
            environment.step(action)
            success, detail = environment.success()
            if success and first_success is None:
                first_success = step
        result.update(hold_steps=steps, hold_first_success_step=first_success, final_checker=detail)
        result["nontrivial_under_hold_control"] = first_success is None
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if environment is not None:
            environment.close()
    (output / f"{task_id}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({
            key: value
            for key, value in result.items()
            if key not in {"initial_state", "initial_checker", "final_checker"}
        }),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--hold-steps", type=int, default=80)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.tasks.glob("*.json"), key=lambda path: int(path.stem))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(inspect, path, args.output, args.hold_steps) for path in paths]
        results = [future.result() for future in futures]
    summary = {
        "task_count": len(results),
        "scene_count": sum(bool(r.get("scene_and_control_ok")) for r in results),
        "checker_count": sum(bool(r.get("checker_implemented")) for r in results),
        "nontrivial_count": sum(bool(r.get("nontrivial_under_hold_control")) for r in results),
        "positive_control_qualified_count": 0,
        "records": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
