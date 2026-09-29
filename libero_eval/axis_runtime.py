"""Headless AXIS MuJoCo runtime used by the validator.

The task API is the source of MJCF, initial state, and checker configuration.
Those mutable responses are accepted only when their hashes match the selected
frozen manifest. Scene assets are downloaded lazily into a local cache.

Checker implementations cover the frozen releases and the separately sourced
historical task import. Unknown types fail closed with an actionable error.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import pathlib
import posixpath
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote


AXIS_V1_NAME = "axis_v1.0"
AXIS_V1_CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs" / "benchmarks" / f"{AXIS_V1_NAME}.yaml"
DEFAULT_MANIFEST = AXIS_V1_CONFIG_PATH
DEFAULT_CACHE = pathlib.Path(__file__).resolve().parents[1] / ".cache" / "axis"
DEFAULT_TASK_SNAPSHOT_ROOT = AXIS_V1_CONFIG_PATH.with_name(f"{AXIS_V1_NAME}-tasks")


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(path: pathlib.Path = DEFAULT_MANIFEST, *, expected_name: str | None = None) -> dict[str, Any]:
    path = pathlib.Path(path)
    if path.suffix in (".yaml", ".yml"):
        manifest = _load_yaml_manifest(path)
    else:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"AXIS manifest must be an object: {path}")
    name = manifest.get("name")
    if (
        not isinstance(name, str)
        or re.fullmatch(r"axis[-_]v[0-9]+(?:\.[0-9]+)?", name) is None
        or manifest.get("status") != "runtime-ready"
        or (expected_name is not None and name != expected_name)
    ):
        raise ValueError(f"unsupported or incomplete AXIS manifest: {path}")
    return manifest


def task_runtime(manifest: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Only observation settings may vary between tasks in a frozen release."""
    overrides = spec.get("runtime_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - {"camera", "wrist_camera", "image_width", "image_height"}:
        raise ValueError("task runtime overrides may only select cameras and render dimensions")
    return {**manifest["runtime"], **overrides}


def task_trial_count(spec: dict[str, Any], requested: int) -> int:
    enabled = spec.get("randomization_enabled")
    if enabled is not None and type(enabled) is not bool:
        raise ValueError("task randomization_enabled must be a boolean")
    if enabled is False:
        return 1
    return requested


def _load_yaml_manifest(path: pathlib.Path) -> dict[str, Any]:
    """Select tasks without duplicating their frozen scene/checker definitions."""
    import yaml

    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid AXIS YAML: {path}: {exc}") from exc
    allowed = {
        "schema_version",
        "name",
        "source_manifest",
        "source_manifest_sha256",
        "protocol_revision",
        "policy_seed",
        "tasks",
    }
    if not isinstance(config, dict) or set(config) != allowed or config["schema_version"] != 1:
        raise ValueError(f"AXIS YAML requires exactly these fields: {sorted(allowed)}: {path}")
    source_name = config["source_manifest"]
    if (
        not isinstance(source_name, str)
        or pathlib.Path(source_name).name != source_name
        or pathlib.Path(source_name).suffix != ".json"
    ):
        raise ValueError("source_manifest must be a JSON filename in the YAML directory")
    source_path = path.with_name(source_name)
    source = load_manifest(source_path)
    if canonical_json_sha256(source) != config["source_manifest_sha256"]:
        raise ValueError(f"source_manifest_sha256 mismatch: {source_path}")
    if not isinstance(config["protocol_revision"], str) or not config["protocol_revision"].strip():
        raise ValueError("AXIS YAML protocol_revision must be a non-empty string")
    if type(config["policy_seed"]) is not int or not 0 <= config["policy_seed"] < 2**32:
        raise ValueError("AXIS YAML policy_seed must be an integer in [0, 2**32)")
    available = task_specs(source)
    selected = task_specs(config)
    tasks = []
    for task_id, task in selected.items():
        if set(task) != {"task_id", "name_zh", "instruction"}:
            raise ValueError(f"AXIS YAML task {task_id} requires task_id, name_zh and instruction")
        if task_id not in available:
            raise ValueError(f"unknown AXIS task id {task_id} in {source_path}")
        if task["instruction"] != available[task_id]["instruction"]:
            raise ValueError(f"AXIS task {task_id} instruction differs from the source manifest")
        if not isinstance(task["name_zh"], str) or not task["name_zh"].strip():
            raise ValueError(f"AXIS task {task_id} name_zh must be a non-empty string")
        tasks.append({**available[task_id], "name_zh": task["name_zh"]})
    return {
        **source,
        "name": config["name"],
        "protocol_revision": config["protocol_revision"],
        "policy_seed": config["policy_seed"],
        "tasks": tasks,
        "task_snapshot_root": f"{source_path.stem}-tasks",
        "subset_provenance": {
            "source_benchmark": source["name"],
            "source_manifest_sha256": config["source_manifest_sha256"],
            "selected_task_ids": list(selected),
        },
    }


def task_specs(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("AXIS manifest tasks must be a non-empty list")
    result = {}
    for task in tasks:
        if not isinstance(task, dict) or type(task.get("task_id")) is not int or task["task_id"] < 1:
            raise ValueError("AXIS task_id must be a positive integer")
        task_id = task["task_id"]
        if task_id in result:
            raise ValueError(f"duplicate AXIS task id {task_id}; the manifest requires unique task ids")
        result[task_id] = task
    return result


def _atomic_write(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_task_payload(payload: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    task_id = int(spec["task_id"])
    if int(payload.get("id", -1)) != task_id:
        raise ValueError(f"AXIS task API returned id={payload.get('id')!r}, expected {task_id}")
    if payload.get("name") != spec["instruction"]:
        raise ValueError(
            f"AXIS task {task_id} name drifted: got {payload.get('name')!r}, expected {spec['instruction']!r}"
        )
    checks = {
        "mjcf_xml": hashlib.sha256(str(payload.get("mjcf_xml") or "").encode("utf-8")).hexdigest(),
        "checker_config": canonical_json_sha256(payload.get("checker_config")),
        "initial_state": canonical_json_sha256(payload.get("initial_state")),
    }
    expected = {
        "mjcf_xml": spec["mjcf_sha256"],
        "checker_config": spec["checker_sha256"],
        "initial_state": spec["initial_state_sha256"],
    }
    drifted = [field for field in checks if checks[field] != expected[field]]
    if drifted:
        detail = ", ".join(f"{field}={checks[field]} expected={expected[field]}" for field in drifted)
        raise ValueError(f"AXIS task {task_id} runtime drifted from the frozen definition: {detail}")
    if not payload.get("mjcf_xml") or not payload.get("checker_config"):
        raise ValueError(f"AXIS task {task_id} has no executable MJCF/checker payload")
    verified = {
        "id": task_id,
        "name": payload["name"],
        "status": payload.get("status"),
        "embodiment": payload.get("embodiment"),
        "mjcf_xml": payload["mjcf_xml"],
        "checker_config": payload["checker_config"],
        "initial_state": payload.get("initial_state"),
    }
    official_hash = spec.get("official_randomization_sha256")
    if official_hash is not None or "official_randomization" in payload:
        config = payload.get("official_randomization")
        if not isinstance(config, dict) or canonical_json_sha256(config) != official_hash:
            raise ValueError(f"AXIS task {task_id} randomization is missing, unpinned or hash-mismatched")
        verified["official_randomization"] = config
    return verified


@dataclass(frozen=True)
class ResolvedTaskPayload:
    data: dict[str, Any]
    source: str
    canonical_sha256: str


def resolve_task_payload(
    spec: dict[str, Any],
    *,
    api_base_url: str,
    selection_contract: int,
    cache_root: pathlib.Path,
    snapshot_root: pathlib.Path | None = DEFAULT_TASK_SNAPSHOT_ROOT,
    refresh: bool = False,
) -> ResolvedTaskPayload:
    """Load a pinned task definition and report its auditable source.

    Release snapshots are preferred over mutable cache/API state. ``refresh``
    is an explicit upstream-drift audit: it bypasses both snapshot and cache,
    but never mutates the release snapshot.
    """

    task_id = int(spec["task_id"])
    cache_path = cache_root / "tasks" / f"{task_id}.json"
    if not refresh and snapshot_root is not None and snapshot_root.is_dir():
        snapshot_path = snapshot_root / f"{task_id}.json"
        if not snapshot_path.is_file():
            raise FileNotFoundError(f"frozen AXIS task snapshot is missing: {snapshot_path}")
        verified = verify_task_payload(json.loads(snapshot_path.read_text(encoding="utf-8")), spec)
        _atomic_write(cache_path, json.dumps(verified, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        return ResolvedTaskPayload(verified, "frozen-snapshot", canonical_json_sha256(verified))

    if cache_path.is_file() and not refresh:
        verified = verify_task_payload(json.loads(cache_path.read_text(encoding="utf-8")), spec)
        return ResolvedTaskPayload(verified, "verified-cache", canonical_json_sha256(verified))

    import httpx

    url = f"{api_base_url.rstrip('/')}/tasks/{task_id}"
    response = httpx.get(url, params={"selection_contract": selection_contract}, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    verified = verify_task_payload(response.json(), spec)
    _atomic_write(cache_path, json.dumps(verified, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return ResolvedTaskPayload(verified, "task-api", canonical_json_sha256(verified))


def fetch_task_payload(
    spec: dict[str, Any],
    *,
    api_base_url: str,
    selection_contract: int,
    cache_root: pathlib.Path,
    snapshot_root: pathlib.Path | None = DEFAULT_TASK_SNAPSHOT_ROOT,
    refresh: bool = False,
) -> dict[str, Any]:
    return resolve_task_payload(
        spec,
        api_base_url=api_base_url,
        selection_contract=selection_contract,
        cache_root=cache_root,
        snapshot_root=snapshot_root,
        refresh=refresh,
    ).data


def normalize_asset_reference(source_path: str, reference: str) -> str:
    if not reference or reference.startswith(("/", "\\")):
        raise ValueError(f"invalid absolute/empty MJCF asset reference: {reference!r}")
    normalized = posixpath.normpath(posixpath.join(posixpath.dirname(source_path), reference.replace("\\", "/")))
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"MJCF asset escapes cache root: source={source_path!r} reference={reference!r}")
    return normalized.lstrip("./")


class AssetCache:
    def __init__(self, root: pathlib.Path, base_url: str, workers: int = 16) -> None:
        self.root = root
        self.base_url = base_url.rstrip("/") + "/"
        self.workers = max(1, int(workers))
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, normalized: str) -> pathlib.Path:
        path = self.root / normalized
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError(f"asset path escapes cache root: {normalized!r}") from exc
        return path

    def _url(self, normalized: str) -> str:
        return self.base_url + "/".join(quote(part, safe="") for part in normalized.split("/"))

    def fetch(self, normalized: str) -> pathlib.Path:
        destination = self.path_for(normalized)
        if destination.is_file() and destination.stat().st_size > 0:
            return destination

        import httpx

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(
                    timeout=httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0),
                    follow_redirects=True,
                    http2=True,
                    limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
                ) as client:
                    response = client.get(self._url(normalized))
                    response.raise_for_status()
                    if not response.content:
                        raise RuntimeError(f"empty AXIS asset response for {normalized}")
                    _atomic_write(destination, response.content)
                    return destination
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"failed to download AXIS asset {normalized}: {last_error}") from last_error

    def fetch_many(self, paths: set[str]) -> None:
        missing = sorted(path for path in paths if not self.path_for(path).is_file())
        if not missing:
            return
        failures: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.workers, len(missing))) as executor:
            futures = {executor.submit(self.fetch, path): path for path in missing}
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{path}: {exc}")
        if failures:
            raise RuntimeError("AXIS scene asset download failed:\n  " + "\n  ".join(failures))

    def prepare_scene(
        self,
        task_id: int,
        mjcf_xml: str,
        *,
        scene_key: str | None = None,
    ) -> tuple[pathlib.Path, dict[str, int]]:
        scene_name = (
            f"axis-task-{task_id}.xml"
            if scene_key is None
            else f"axis-task-{task_id}-{hashlib.sha256(scene_key.encode('utf-8')).hexdigest()[:16]}.xml"
        )
        scene_path = self.root / "scenes" / scene_name
        _atomic_write(scene_path, mjcf_xml.encode("utf-8"))
        entry_name = scene_path.relative_to(self.root).as_posix()

        xml_queue: list[tuple[str, str]] = [(entry_name, mjcf_xml)]
        visited_xml: set[str] = set()
        binary_paths: set[str] = set()
        while xml_queue:
            source_name, source_xml = xml_queue.pop()
            if source_name in visited_xml:
                continue
            visited_xml.add(source_name)
            root = ET.fromstring(source_xml)
            for node in root.iter():
                reference = node.get("file")
                if not reference:
                    continue
                normalized = normalize_asset_reference(source_name, reference)
                is_xml = node.tag == "include" or pathlib.PurePosixPath(normalized).suffix.lower() in {".xml", ".mjcf"}
                if is_xml:
                    dependency = self.fetch(normalized)
                    xml_queue.append((normalized, dependency.read_text(encoding="utf-8")))
                else:
                    binary_paths.add(normalized)
        self.fetch_many(binary_paths)
        return scene_path, {"xml_files": len(visited_xml), "binary_files": len(binary_paths)}


JOINT_NAMES = (
    "franka/panda_finger_joint1",
    "franka/panda_finger_joint2",
    "franka/panda_joint1",
    "franka/panda_joint2",
    "franka/panda_joint3",
    "franka/panda_joint4",
    "franka/panda_joint5",
    "franka/panda_joint6",
    "franka/panda_joint7",
)


def _named_id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return object_id


def _apply_initial_state(mujoco: Any, model: Any, data: Any, initial_state: dict[str, Any] | None) -> None:
    if not initial_state:
        mujoco.mj_forward(model, data)
        return

    def apply_entity(entity_key: str, config: dict[str, Any]) -> None:
        position = config.get("pos")
        rotation = config.get("rot")
        if position or rotation:
            for body_name in (f"{entity_key}/", entity_key):
                body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
                if body_id < 0:
                    continue
                for joint_id in range(model.njnt):
                    if int(model.jnt_bodyid[joint_id]) != body_id:
                        continue
                    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                        continue
                    address = int(model.jnt_qposadr[joint_id])
                    if position and len(position) >= 3:
                        data.qpos[address : address + 3] = position[:3]
                    if rotation and len(rotation) >= 4:
                        data.qpos[address + 3 : address + 7] = rotation[:4]
                    break
                break

        dof_positions = config.get("dof_pos") or config.get("dofPos") or {}
        for joint_key, value in dof_positions.items():
            candidates = (
                [joint_key]
                if "/" in joint_key
                else [
                    f"franka/{joint_key}" if entity_key == "franka" else "",
                    f"{entity_key}/{joint_key}",
                    f"{entity_key}_base/{joint_key}",
                    joint_key,
                ]
            )
            for candidate in candidates:
                if not candidate:
                    continue
                joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate))
                if joint_id >= 0 and model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                    data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)
                    break

    for section in ("objects", "robots"):
        for entity_key, config in (initial_state.get(section) or {}).items():
            apply_entity(entity_key, config or {})
    mujoco.mj_forward(model, data)


@dataclass
class RuntimeState:
    positions: dict[str, list[float]]
    orientations: dict[str, list[float]]
    joints: dict[str, float]
    site_positions: dict[str, list[float]] = field(default_factory=dict)
    site_orientations: dict[str, list[float]] = field(default_factory=dict)


def _runtime_state(mujoco: Any, model: Any, data: Any) -> RuntimeState:
    positions: dict[str, list[float]] = {}
    orientations: dict[str, list[float]] = {}
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name or name == "world":
            continue
        positions[name] = [float(value) for value in data.xpos[body_id]]
        w, x, y, z = [float(value) for value in data.xquat[body_id]]
        orientations[name] = [x, y, z, w]
    joints: dict[str, float] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name and model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            joints[name] = float(data.qpos[int(model.jnt_qposadr[joint_id])])
    import numpy as np

    site_positions, site_orientations = {}, {}
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if not name:
            continue
        site_positions[name] = [float(value) for value in data.site_xpos[site_id]]
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
        w, x, y, z = map(float, quaternion)
        site_orientations[name] = [x, y, z, w]
    return RuntimeState(positions, orientations, joints, site_positions, site_orientations)


def _resolve(name: str, mapping: dict[str, Any]) -> Any | None:
    if name in mapping:
        return mapping[name]
    slash_suffix = "/" + name
    for key, value in mapping.items():
        if key.endswith(slash_suffix):
            return value
    for key, value in mapping.items():
        if key.endswith(name):
            return value
    return None


def _quat_normalize(quaternion: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    return [value / norm for value in quaternion] if norm >= 1e-12 else [0.0, 0.0, 0.0, 1.0]


def _quat_multiply(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quat_rotate(quaternion: list[float], vector: list[float]) -> list[float]:
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx, ty, tz = 2 * (qy * vz - qz * vy), 2 * (qz * vx - qx * vz), 2 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    ]


def evaluate_checker(
    config: dict[str, Any],
    current: RuntimeState,
    runtime_initial: RuntimeState,
) -> tuple[bool, dict[str, Any]]:
    checker_type = config.get("type")
    if checker_type == "CompositeChecker":
        children = [evaluate_checker(child, current, runtime_initial) for child in config.get("checkers") or []]
        operator = str(config.get("operator", "AND")).upper()
        values = [passed for passed, _ in children]
        if operator == "OR":
            passed = any(values)
        elif operator == "NOT":
            passed = len(values) == 1 and not values[0]
        else:
            passed = bool(values) and all(values)
        return passed, {
            "checker_type": checker_type,
            "operator": operator,
            "passed": passed,
            "sub_results": [detail for _, detail in children],
        }

    if checker_type in {"RelativeCylinderChecker", "RelativePositionBoundsChecker"}:
        object_name, reference_name = str(config.get("objName", "")), str(config.get("refName", ""))
        object_position = _resolve(object_name, current.positions)
        reference_type = config.get("refType", "body")
        if reference_type not in {"body", "object", "site"}:
            raise ValueError(f"unsupported reference type {reference_type!r}")
        reference_position = _resolve(
            reference_name, current.site_positions if reference_type == "site" else current.positions
        )
        if object_position is None or reference_position is None:
            return False, {"checker_type": checker_type, "passed": False, "reason": "missing positions"}
        delta = [object_position[index] - reference_position[index] for index in range(3)]
        if checker_type == "RelativeCylinderChecker":
            radius = float(config.get("xyRadius", 0.06))
            minimum, maximum = float(config.get("heightMin", 0.0)), float(config.get("heightMax", 0.03))
            distance = math.hypot(delta[0], delta[1])
            passed = distance < radius and minimum < delta[2] < maximum
            return passed, {
                "checker_type": checker_type,
                "passed": passed,
                "xy_distance": distance,
                "xy_radius": radius,
                "dz": delta[2],
                "height_min": minimum,
                "height_max": maximum,
            }
        ranges = [config.get("xRange"), config.get("yRange"), config.get("zRange")]
        checks = [bounds[0] < delta[index] < bounds[1] for index, bounds in enumerate(ranges) if bounds is not None]
        passed = bool(checks) and all(checks)
        return passed, {"checker_type": checker_type, "passed": passed, "delta": delta, "ranges": ranges}

    if checker_type == "GripperOpenChecker":
        threshold = float(config.get("threshold", 0.2))
        finger_values = {
            name: value
            for name, value in current.joints.items()
            if "finger" in name.lower() or "gripper" in name.lower()
        }
        passed = any(abs(value) + 1e-4 >= threshold for value in finger_values.values())
        return passed, {
            "checker_type": checker_type,
            "passed": passed,
            "threshold": threshold,
            "joint_values": finger_values,
        }

    if checker_type == "DirectedRotationChecker":
        body_name = str(config.get("bodyName", "sample/"))
        current_quaternion = _resolve(body_name, current.orientations)
        initial_quaternion = _resolve(body_name, runtime_initial.orientations)
        if current_quaternion is None or initial_quaternion is None:
            return False, {"checker_type": checker_type, "passed": False, "reason": "missing orientation"}
        current_q, initial_q = _quat_normalize(current_quaternion), _quat_normalize(initial_quaternion)
        local_up = [float(value) for value in config.get("localUprightAxis", [0.0, 0.0, 1.0])]
        initial_up, current_up = _quat_rotate(initial_q, local_up), _quat_rotate(current_q, local_up)
        up_norms = math.sqrt(sum(value * value for value in initial_up)) * math.sqrt(
            sum(value * value for value in current_up)
        )
        up_dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(initial_up, current_up)) / up_norms))
        tilt = math.degrees(math.acos(up_dot))
        relative = _quat_normalize(
            _quat_multiply(current_q, [-initial_q[0], -initial_q[1], -initial_q[2], initial_q[3]])
        )
        axis_norm = math.sqrt(sum(value * value for value in initial_up)) or 1.0
        axis = [value / axis_norm for value in initial_up]
        scalar = sum(relative[index] * axis[index] for index in range(3))
        angle = ((math.degrees(2.0 * math.atan2(scalar, relative[3])) + 180.0) % 360.0) - 180.0
        target = float(config["targetAngleDeg"])
        error = abs(((angle - target + 180.0) % 360.0) - 180.0)
        tolerance = float(config.get("angleToleranceDeg", 20.0))
        max_tilt = config.get("maxTiltDeg")
        passed = error <= tolerance and (max_tilt is None or tilt <= float(max_tilt))
        return passed, {
            "checker_type": checker_type,
            "passed": passed,
            "body_name": body_name,
            "twist_angle_deg": angle,
            "target_angle_deg": target,
            "angle_error_deg": error,
            "tilt_angle_deg": tilt,
        }

    return _evaluate_legacy_checker(config, current, runtime_initial)


def _evaluate_legacy_checker(config: dict, current: RuntimeState, initial: RuntimeState) -> tuple[bool, dict]:
    """Historical frontend geometry: body quaternions xyzw; drawer relativeQuat wxyz.

    Configuration key aliases are normalized at import time. Missing named state
    never succeeds; unsupported or malformed predicates raise rather than score.
    """
    kind = config.get("type")
    detail = {"checker_type": kind, "passed": False}

    def result(passed, **values):
        return bool(passed), {**detail, **values, "passed": bool(passed)}

    if kind in {"FrameBBoxChecker", "DrawerBBoxChecker"}:
        obj = _resolve(str(config["objName"]), current.positions)
        if kind == "FrameBBoxChecker":
            frame_type = config.get("frameType", "body")
            if frame_type not in {"body", "object", "site"}:
                raise ValueError(f"unsupported frame type {frame_type!r}")
            sites = frame_type == "site"
            name = str(config["frameName"])
            origin = _resolve(name, current.site_positions if sites else current.positions)
            quat = _resolve(name, current.site_orientations if sites else current.orientations)
            lower, upper = config["lower"], config["upper"]
            tolerance = 0.0
        else:
            name = str(config["cabinetName"])
            origin = _resolve(name, current.positions)
            quat = _resolve(name, current.orientations)
            prefix = name.rstrip("/") + "/"
            joint = config.get("jointName")
            if joint:
                value = current.joints.get(joint if "/" in joint else prefix + joint)
            else:
                joints = [v for k, v in current.joints.items() if k.startswith(prefix)]
                index = int(config.get("jointIndex", 0))
                value = joints[index] if 0 <= index < len(joints) else None
            if origin is None or quat is None or value is None:
                return result(False, reason="missing cabinet pose or drawer joint")
            offset = [float(v) for v in config.get("baseOffset", [0, 0, 0])]
            axis = int(config.get("displacementAxis", 1))
            if axis not in (0, 1, 2):
                raise ValueError("drawer displacementAxis must be 0, 1, or 2")
            offset[axis] += value
            origin = [a + b for a, b in zip(origin, _quat_rotate(_quat_normalize(quat), offset))]
            w, x, y, z = config.get("relativeQuat", [1, 0, 0, 0])
            quat = _quat_multiply(_quat_normalize(quat), _quat_normalize([x, y, z, w]))
            half = config["halfSize"]
            lower, upper = [-v for v in half], half
            tolerance = 1e-6
        if obj is None or origin is None or quat is None:
            return result(False, reason="missing object or frame pose")
        x, y, z, w = _quat_normalize(quat)
        local = _quat_rotate([-x, -y, -z, w], [a - b for a, b in zip(obj, origin)])
        passed = all(lo - tolerance <= v <= hi + tolerance for v, lo, hi in zip(local, lower, upper))
        return result(passed, local_position=local, lower=lower, upper=upper)

    if kind == "JointThresholdChecker":
        name = str(config["jointName"])
        if "/" not in name and config.get("objName"):
            name = str(config["objName"]).rstrip("/") + "/" + name
        value = current.joints.get(name)
        mode, threshold = config.get("mode", "ge"), float(config.get("threshold", 0))
        if mode not in {"gt", "ge", "lt", "le"}:
            raise ValueError(f"unsupported joint threshold mode {mode!r}")
        if value is None:
            return result(False, reason="missing joint", joint_name=name)
        passed = {"gt": value > threshold, "ge": value >= threshold, "lt": value < threshold, "le": value <= threshold}[
            mode
        ]
        return result(passed, joint_name=name, value=value, mode=mode, threshold=threshold)

    if kind in {"SamplePositionDeltaChecker", "BowlPositionChecker"}:
        name = (
            config.get("sampleBodyName", "sample/")
            if kind.startswith("Sample")
            else config.get("bowlBodyName", "bowl/")
        )
        position = _resolve(str(name), current.positions)
        if position is None:
            return result(False, reason="missing object position", body_name=name)
        if kind == "BowlPositionChecker":
            if config.get("minBounds") is not None and config.get("maxBounds") is not None:
                lower, upper = config["minBounds"], config["maxBounds"]
                lower = [lower[k] for k in "xyz"] if isinstance(lower, dict) else lower
                upper = [upper[k] for k in "xyz"] if isinstance(upper, dict) else upper
                return result(all(lo <= v <= hi for lo, v, hi in zip(lower, position, upper)), position=position)
            if config.get("positionThreshold") is not None:
                distance = math.sqrt(sum(v * v for v in position))
                return result(distance <= float(config["positionThreshold"]), distance=distance)
            raise ValueError("BowlPositionChecker requires bounds or a distance threshold")
        if kind.startswith("Sample"):
            origin = config.get("initialPosition")
            if origin is None:
                origin = _resolve(str(name), initial.positions)
            if origin is None:
                return result(False, reason="missing initial position")
            position = [a - b for a, b in zip(position, origin)]
            axes = config.get("axes", ["x", "z"])
            axes = axes.split(",") if isinstance(axes, str) else axes
        else:
            axes = ["x", "y", "z"]
        checks = []
        for axis in axes:
            axis = axis.strip().lower()
            if axis not in "xyz" or len(axis) != 1:
                raise ValueError(f"invalid position axis {axis!r}")
            value = position["xyz".index(axis)]
            for bound, compare in (("min", lambda a, b: a >= b), ("max", lambda a, b: a <= b)):
                key = bound + "Delta" + axis.upper()
                if config.get(key) is not None:
                    checks.append(compare(value, float(config[key])))
        if not checks:
            raise ValueError(f"{kind} requires at least one position bound")
        return result(all(checks), position=position)

    if kind == "SampleRotationChecker":
        name = str(config.get("sampleBodyName", "sample/"))
        current_q = _resolve(name, current.orientations)
        initial_q = (
            _resolve(name, initial.orientations)
            if config.get("captureRuntimeInitial", True)
            else config.get("initialRotation", [0, 0, 0, 1])
        )
        if current_q is None or initial_q is None:
            return result(False, reason="missing orientation")
        current_q, initial_q = _quat_normalize(current_q), _quat_normalize(initial_q)
        if config.get("tiltOnly", True):
            up, initial_up = _quat_rotate(current_q, [0, 0, 1]), _quat_rotate(initial_q, [0, 0, 1])
            dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(up, initial_up))))
            angle = math.degrees(math.acos(dot))
            direction = config.get("tiltWorldDirection")
            if direction is not None:
                direction = [direction[k] for k in "xyz"] if isinstance(direction, dict) else direction
                projected = sum(a * b for a, b in zip(direction, initial_up))
                planar = [a - projected * b for a, b in zip(direction, initial_up)]
                norm = math.sqrt(sum(v * v for v in planar))
                if norm >= 1e-8:
                    angle = math.degrees(math.atan2(sum(a * b / norm for a, b in zip(up, planar)), dot))
        else:
            x, y, z, w = initial_q
            relative = _quat_multiply(current_q, [-x, -y, -z, w])
            angle = math.degrees(2 * math.acos(min(1.0, abs(relative[3]))))
        threshold = float(config.get("tipAngleThreshold", 30))
        return result(angle >= threshold, tip_angle_deg=angle, threshold_deg=threshold)

    raise ValueError(f"AXIS runtime does not implement checker type {kind!r}")


class AxisEnvironment:
    def __init__(self, scene_path: pathlib.Path, payload: dict[str, Any], runtime: dict[str, Any]) -> None:
        import mujoco
        import numpy as np

        self.mujoco = mujoco
        self.np = np
        self.payload = payload
        self.runtime = runtime
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.model.opt.disableflags = int(self.model.opt.disableflags) | int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        self.data = mujoco.MjData(self.model)
        self.control_period_s = float(runtime["control_period_s"])
        exact_steps = self.control_period_s / float(self.model.opt.timestep)
        self.steps_per_control = int(round(exact_steps))
        if self.steps_per_control < 1 or abs(exact_steps - self.steps_per_control) > 1e-9:
            raise ValueError(
                f"AXIS control period {self.control_period_s} is not divisible by "
                f"MJCF timestep {self.model.opt.timestep}"
            )
        self.joint_ids = [
            _named_id(mujoco, self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in runtime["observation_joint_order"]
        ]
        self.qpos_indices = [int(self.model.jnt_qposadr[joint_id]) for joint_id in self.joint_ids]
        self.actuator_ids = [
            _named_id(mujoco, self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in runtime["observation_joint_order"]
        ]
        self.render_width = int(runtime.get("image_width", runtime["image_size"]))
        self.render_height = int(runtime.get("image_height", runtime["image_size"]))
        self.axis_randomizer = None
        self.axis_scene = None
        self.reset_randomization: dict[str, Any] | None = None
        if "official_randomization" in payload:
            from axis_perturbations import AxisRandomizer

            config = payload["official_randomization"]
            visual = config.get("visual")
            use_axis_scene = isinstance(visual, dict) and visual.get("mode") in {
                "official_franka_v6",
                "official_franka_components",
                "official_franka_components_v2",
            }
            if use_axis_scene:
                config = {**config, "visual": None}
            self.axis_randomizer = AxisRandomizer(
                self.model,
                config,
                asset_root=scene_path.parent.parent,
                task_id=int(payload["id"]),
                task_name=payload["name"],
                width=self.render_width,
                height=self.render_height,
            )
            if use_axis_scene:
                from axis_scene import AxisSceneRenderer

                self.axis_scene = AxisSceneRenderer(
                    scene_path,
                    payload,
                    self.model,
                    width=self.render_width,
                    height=self.render_height,
                )
        self.renderer = (
            self.axis_scene.renderer
            if self.axis_scene is not None
            else mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        )
        self.runtime_initial: RuntimeState | None = None

    def close(self) -> None:
        self.renderer.close()

    def reset(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        _apply_initial_state(self.mujoco, self.model, self.data, self.payload.get("initial_state"))
        if self.axis_randomizer is not None:
            self.reset_randomization = self.axis_randomizer.apply(self.data, self.renderer)
        self.hold_current_pose()
        for _ in range(int(self.runtime["settle_control_steps"])):
            self.mujoco.mj_step(self.model, self.data, nstep=self.steps_per_control)
        self.runtime_initial = _runtime_state(self.mujoco, self.model, self.data)
        if self.axis_randomizer is not None:
            if not self.np.isfinite(self.data.qpos).all() or not self.np.isfinite(self.data.qvel).all():
                raise ValueError("AXIS randomized reset produced a non-finite state")
            if self.np.any(self.data.warning.number):
                raise ValueError("AXIS randomized reset produced a MuJoCo warning")
            passed, detail = self.success()
            if passed:
                raise ValueError(f"AXIS randomized reset already satisfies the checker: {detail}")
        if self.axis_scene is not None:
            self.reset_randomization["visual"] = self.axis_scene.reset(self.data)

    def hold_current_pose(self) -> None:
        for actuator_id, qpos_index in zip(self.actuator_ids, self.qpos_indices):
            self.data.ctrl[actuator_id] = self.data.qpos[qpos_index]

    def observation_state(self) -> Any:
        return self.np.asarray([self.data.qpos[index] for index in self.qpos_indices], dtype=self.np.float32)

    def render(self, *, camera: str | None = None) -> Any:
        if self.axis_scene is not None:
            return self.axis_scene.render(
                self.data,
                str(camera if camera is not None else self.runtime["camera"]),
            )
        self.renderer.update_scene(self.data, camera=str(camera if camera is not None else self.runtime["camera"]))
        return self.renderer.render().copy()

    def step(self, action: Any) -> None:
        values = self.np.asarray(action, dtype=self.np.float64).reshape(-1)
        if values.shape != (len(self.actuator_ids),) or not self.np.isfinite(values).all():
            raise ValueError(f"AXIS policy action must contain {len(self.actuator_ids)} finite joint targets")
        for index, (actuator_id, joint_id) in enumerate(zip(self.actuator_ids, self.joint_ids)):
            value = float(values[index])
            if bool(self.model.jnt_limited[joint_id]):
                lower, upper = self.model.jnt_range[joint_id]
                value = min(float(upper), max(float(lower), value))
            self.data.ctrl[actuator_id] = value
        self.mujoco.mj_step(self.model, self.data, nstep=self.steps_per_control)

    def success(self) -> tuple[bool, dict[str, Any]]:
        if self.runtime_initial is None:
            raise RuntimeError("AXIS environment must be reset before checker evaluation")
        root = self.payload["checker_config"].get("checker", self.payload["checker_config"])
        return evaluate_checker(root, _runtime_state(self.mujoco, self.model, self.data), self.runtime_initial)
