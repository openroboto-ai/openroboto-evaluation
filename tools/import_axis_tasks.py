#!/usr/bin/env python3
"""Import the pinned AXIS task source and record runtime compatibility adjustments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pathlib
import sys
import xml.etree.ElementTree as ET


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))
from axis_runtime import canonical_json_sha256  # noqa: E402


ALIASES = {
    "obj_name": "objName",
    "ref_name": "refName",
    "ref_type": "refType",
    "xy_radius": "xyRadius",
    "height_min": "heightMin",
    "height_max": "heightMax",
    "frame_name": "frameName",
    "frame_type": "frameType",
    "cabinet_name": "cabinetName",
    "half_size": "halfSize",
    "joint_name": "jointName",
    "joint_index": "jointIndex",
    "base_offset": "baseOffset",
    "relative_quat": "relativeQuat",
    "displacement_axis": "displacementAxis",
}
READY = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.785398,
    "panda_joint3": 0.0,
    "panda_joint4": -2.356194,
    "panda_joint5": 0.0,
    "panda_joint6": 1.570796,
    "panda_joint7": 0.785398,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}


def normalize_checker(value):
    if isinstance(value, list):
        return [normalize_checker(child) for child in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, child in value.items():
        target = ALIASES.get(key, key)
        normalized = normalize_checker(child)
        if target in result and result[target] != normalized:
            raise ValueError(f"conflicting checker aliases: {target}")
        result[target] = normalized
    return result


def parse_object(value, field):
    result = json.loads(value) if isinstance(value, str) else copy.deepcopy(value)
    if result is not None and not isinstance(result, dict):
        raise ValueError(f"legacy {field} must decode to an object or null")
    return result


def normalize_scene(xml):
    tree = ET.fromstring(xml)
    cameras = list(tree.iter("camera"))
    changes = ["Canonicalized XML whitespace and attribute ordering; physics parameters retained."]
    for body in tree.iter("body"):
        if body.get("name") != "flat_stove/":
            continue
        position = [float(value) for value in body.get("pos", "0 0 0").split()]
        if len(position) != 3 or abs(position[2] + 0.007) > 1e-8:
            raise ValueError("stove placement no longer matches the reviewed ground-intersection correction")
        position[2] = 0.0
        body.set("pos", " ".join(map(str, position)))
        changes.append(
            "Benchmark scene correction: raised the fixed stove root by 7 mm, from z=-0.007 to z=0. "
            "Its rotating button intersected the z=0 ground by 6.64 mm and generated singular contact forces; "
            "robot, joint limits, object geometry and relative success predicates are retained."
        )
    if not any(camera.get("name") == "camera0" for camera in cameras):
        if cameras:
            old_name = cameras[0].get("name")
            cameras[0].set("name", "camera0")
            changes.append(f"Renamed source camera {old_name} to camera0; pose retained.")
        else:
            # Historical web tasks used the viewer camera. Freeze an explicit common
            # head camera for this benchmark version instead of relying on GUI state.
            position, target = (1.4, -1.2, 1.0), (0.45, 0.05, 0.15)
            z = [a - b for a, b in zip(position, target)]
            norm = math.sqrt(sum(v * v for v in z))
            z = [v / norm for v in z]
            x = [-z[1], z[0], 0.0]
            norm = math.sqrt(sum(v * v for v in x))
            x = [v / norm for v in x]
            y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
            world = tree.find("worldbody")
            if world is None:
                raise ValueError("legacy scene has no worldbody")
            ET.SubElement(
                world,
                "camera",
                name="camera0",
                pos=" ".join(map(str, position)),
                xyaxes=" ".join(map(str, x + y)),
                fovy="45",
            )
            changes.append("Added a frozen validator camera; the legacy XML had no explicit camera.")
    for node in tree.iter():
        if node.text is not None and not node.text.strip():
            node.text = None
        if node.tail is not None and not node.tail.strip():
            node.tail = None
        attributes = dict(sorted(node.attrib.items()))
        node.attrib.clear()
        node.attrib.update(attributes)
    return ET.tostring(tree, encoding="unicode"), changes


def normalize_task(task):
    checker = normalize_checker(parse_object(task["checker_config"], "checker_config"))
    if not checker:
        raise ValueError(f"task {task['id']} has no checker")
    state = parse_object(task.get("initial_state"), "initial_state") or {}
    xml, changes = normalize_scene(task["mjcf_xml"])
    if not state.get("robots"):
        state["robots"] = {"franka": {"dof_pos": copy.deepcopy(READY)}}
        changes.append("Made the upstream viewer's fallback Franka ready pose explicit; object states retained.")
    changes.append("Decoded JSON-string fields and canonicalized documented snake/camel checker key aliases.")
    # Two upstream predicates address drawers by an incorrect positional index.
    # The embedded MJCF declares top, middle, bottom; the task text and bounding
    # box height identify the intended drawer independently. Pin its name and
    # retain this benchmark correction in the import audit, never in raw source.
    if task["id"] in {49, 55}:
        root = checker.get("checker", checker)
        expected_index, joint_name = {
            49: (0, "white_cabinet/bottom_level"),
            55: (2, "white_cabinet/top_level"),
        }[task["id"]]
        if root.get("type") != "DrawerBBoxChecker" or root.get("jointIndex") != expected_index:
            raise ValueError(f"task {task['id']} no longer matches the reviewed drawer-index correction")
        del root["jointIndex"]
        root["jointName"] = joint_name
        changes.append(
            f"Benchmark predicate correction: replaced source jointIndex={expected_index} with {joint_name}. "
            "Source index selects the opposite drawer in its embedded MJCF; "
            "title and box height agree on the named drawer."
        )
    return {
        "id": task["id"],
        "name": task["name"],
        "embodiment": "franka",
        "mjcf_xml": xml,
        "checker_config": checker,
        "initial_state": state,
    }, changes


def metadata(task):
    title = task["name"].lower()
    changes_joint = title.startswith(("open", "close", "turn")) or "and close" in title
    compound = " and " in title
    if changes_joint:
        skill = "articulation-composition" if compound else "articulation"
    elif "drawer" in title:
        skill = "drawer-placement"
    elif "wine rack" in title:
        skill = "rack-placement"
    elif title.startswith("grab"):
        skill = "grasp-lift"
    elif title.startswith("water"):
        skill = "tilt-and-position"
    else:
        skill = "support"
    return {
        "skill": skill,
        "horizon": "medium" if compound or title.startswith("water") else "short",
        "horizon_source": "instruction_step_estimate_no_numeric_trajectory_length",
        "numeric_trajectory_length_available": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, default=ROOT / "configs/axis-v1.0")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.source / "source.json").read_text())
    raw = (args.source / "tasks_config.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("AXIS task source differs from the pinned digest")
    tasks = json.loads(raw)["tasks"]
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate AXIS task IDs")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "tasks").mkdir()
    report = {"source": source, "status": "imported-awaiting-qualification", "tasks": [], "excluded": []}
    for task in sorted(tasks, key=lambda t: t["id"]):
        if task["id"] == 23:
            report["excluded"].append({
                "task_id": 23,
                "reason": "Keyboard tutorial; key presses are not a policy action-space task.",
            })
            continue
        payload, changes = normalize_task(task)
        path = output / "tasks" / f"{task['id']}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        report["tasks"].append({
            "task_id": task["id"],
            "instruction": task["name"],
            **metadata(task),
            "mjcf_sha256": hashlib.sha256(payload["mjcf_xml"].encode()).hexdigest(),
            "checker_sha256": canonical_json_sha256(payload["checker_config"]),
            "initial_state_sha256": canonical_json_sha256(payload["initial_state"]),
            "source_task_sha256": canonical_json_sha256(task),
            "compatibility_changes": changes,
            "qualification_complete": False,
        })
    (output / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"imported": len(report["tasks"]), "excluded": report["excluded"], "qualified": 0}))


if __name__ == "__main__":
    main()
