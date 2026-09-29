"""Unmodified AXIS XML helper definitions; imports isolated from upstream services."""
from __future__ import annotations
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from .table_arena_adapter import TableArena

ARENA_PREFIX = "robosuite_arena_"


XML_FILE_ATTR_TAGS = {"include", "mesh", "texture", "hfield"}


DEFAULT_TABLE_TOP_Z = 0.48


DEFAULT_TABLE_CENTER_XY = (0.45, -0.1)


DEFAULT_TABLE_FULL_SIZE = (1.4, 1.2, 0.05)


TABLE_TOP_MARGIN = 0.25


OBJECT_BODY_NAME_NEEDLES = (
    "apple",
    "banana",
    "bowl",
    "grape",
    "lemon",
    "orange",
    "pear",
    "walnut",
)


ARM_ROBOT_BODY_NEEDLES = ("franka", "panda", "ur10", "ur10e", "ur5", "ur5e", "ur_")


BOOSTER_BODY_NEEDLES = ("booster", "trunk", "t1_7dof")


@dataclass(frozen=True)
class ScenePlacement:
    policy: str
    table_center_xy: tuple[float, float]
    table_top_z: float
    table_full_size: tuple[float, float, float]
    preserve_source_table: bool
    arm_scene_z_offset: float
    free_joint_qpos_z_offset: float
    table_surface_geom_names: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _MJCFDocument:
    """An entry MJCF document or one of its recursively included documents."""

    root: ET.Element
    path: Path | None


def _camera_names_from_config(camera_config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(camera["name"]) for camera in camera_config["cameras"])


def _format_float_sequence(values: np.ndarray | list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _camera_xyaxes(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-8:
        raise ValueError("Camera position and target must be different.")
    forward /= norm
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-8:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
    right /= right_norm
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1e-8)
    return np.concatenate([right, up])


def _camera_fovy(camera_config: dict[str, Any]) -> float:
    intrinsics = camera_config.get("intrinsics") or {}
    return float(intrinsics.get("fovy", 58.0))


def _table_relative_camera_pose(camera_config: dict[str, Any], placement: ScenePlacement) -> tuple[np.ndarray, np.ndarray]:
    extrinsics = camera_config.get("extrinsics") or {}
    position_offset = _parse_vec(
        _format_float_sequence(extrinsics.get("position_offset", [1.2, 0.0, 0.7])),
        length=3,
    )
    target_offset = _parse_vec(
        _format_float_sequence(extrinsics.get("target_offset", [0.0, 0.0, 0.12])),
        length=3,
    )
    base = np.asarray([placement.table_center_xy[0], placement.table_center_xy[1], placement.table_top_z], dtype=np.float64)
    return base + position_offset, base + target_offset


def _normalize_quat_wxyz(quat: np.ndarray, *, label: str) -> np.ndarray:
    normalized = np.asarray(quat, dtype=np.float64)
    if normalized.shape != (4,):
        raise ValueError(f"{label} must contain four wxyz values.")
    norm = float(np.linalg.norm(normalized))
    if norm <= 1e-12:
        raise ValueError(f"{label} must be non-zero.")
    return normalized / norm


def _quat_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quat_rotate_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat, label="rotation quaternion")
    vector_quat = np.concatenate([np.zeros(1, dtype=np.float64), np.asarray(vector, dtype=np.float64)])
    conjugate = quat * np.asarray([1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    return _quat_multiply_wxyz(_quat_multiply_wxyz(quat, vector_quat), conjugate)[1:]


WRIST_CAMERA_RANDOMIZATION_MODE = "optical_pose_box_v1"


FRONT_CAMERA_RANDOMIZATION_MODE = "base_table_sector_lookat_v1"


def _strict_object_keys(
    value: Any,
    *,
    label: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    optional = optional or set()
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}.")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {unknown}.")
    return value


def _finite_float(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{label} must be finite.")
    if minimum is not None and resolved < minimum:
        raise ValueError(f"{label} must be >= {minimum}.")
    if maximum is not None and resolved > maximum:
        raise ValueError(f"{label} must be <= {maximum}.")
    return resolved


def _configured_fovy_randomization(
    value: Any,
    *,
    label: str,
    configured_fovy: float,
) -> dict[str, float | str]:
    profile = _strict_object_keys(
        value,
        label=label,
        required={"center", "half_range", "distribution"},
    )
    if str(profile["distribution"]) != "uniform_symmetric":
        raise ValueError(f"{label}.distribution must be 'uniform_symmetric'.")
    center = _finite_float(profile["center"], label=f"{label}.center", minimum=1e-6, maximum=179.999)
    half_range = _finite_float(
        profile["half_range"],
        label=f"{label}.half_range",
        minimum=0.0,
        maximum=89.0,
    )
    if center - half_range <= 0.0 or center + half_range >= 180.0:
        raise ValueError(f"{label} must stay strictly inside (0, 180) degrees.")
    if not math.isclose(center, configured_fovy, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{label}.center={center} must equal the camera intrinsics fovy={configured_fovy}."
        )
    return {
        "center": center,
        "half_range": half_range,
        "distribution": "uniform_symmetric",
    }


def _configured_wrist_camera_randomization(
    camera: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    label = f"camera {camera.get('name')!r} randomization"
    _strict_object_keys(
        raw,
        label=label,
        required={
            "enabled",
            "mode",
            "position_half_ranges_m",
            "orientation_half_ranges_deg",
            "fovy_deg",
        },
    )
    if str((camera.get("mount") or {}).get("type") or "") != "body":
        raise ValueError(f"{label} mode {WRIST_CAMERA_RANDOMIZATION_MODE!r} requires a body-mounted camera.")
    position_raw = _strict_object_keys(
        raw["position_half_ranges_m"],
        label=f"{label}.position_half_ranges_m",
        required={"lateral", "vertical", "depth", "distribution"},
    )
    if str(position_raw["distribution"]) != "independent_uniform_symmetric":
        raise ValueError(
            f"{label}.position_half_ranges_m.distribution must be "
            "'independent_uniform_symmetric'."
        )
    orientation_raw = _strict_object_keys(
        raw["orientation_half_ranges_deg"],
        label=f"{label}.orientation_half_ranges_deg",
        required={"roll", "pitch", "yaw", "distribution", "composition"},
    )
    if str(orientation_raw["distribution"]) != "independent_uniform_symmetric":
        raise ValueError(
            f"{label}.orientation_half_ranges_deg.distribution must be "
            "'independent_uniform_symmetric'."
        )
    if str(orientation_raw["composition"]) != "camera_dr_basis_rz_yaw_ry_pitch_rx_roll_postmultiply":
        raise ValueError(
            f"{label}.orientation_half_ranges_deg.composition has an unsupported value."
        )
    configured_fovy = _camera_fovy(camera)
    return {
        "enabled": bool(raw["enabled"]),
        "mode": WRIST_CAMERA_RANDOMIZATION_MODE,
        "position_half_ranges_m": {
            axis: _finite_float(
                position_raw[axis],
                label=f"{label}.position_half_ranges_m.{axis}",
                minimum=0.0,
            )
            for axis in ("lateral", "vertical", "depth")
        }
        | {"distribution": "independent_uniform_symmetric"},
        "orientation_half_ranges_deg": {
            axis: _finite_float(
                orientation_raw[axis],
                label=f"{label}.orientation_half_ranges_deg.{axis}",
                minimum=0.0,
                maximum=45.0,
            )
            for axis in ("roll", "pitch", "yaw")
        }
        | {
            "distribution": "independent_uniform_symmetric",
            "composition": "camera_dr_basis_rz_yaw_ry_pitch_rx_roll_postmultiply",
        },
        "fovy_deg": _configured_fovy_randomization(
            raw["fovy_deg"],
            label=f"{label}.fovy_deg",
            configured_fovy=configured_fovy,
        ),
        "coordinate_convention": {
            "camera_dr_basis": ["depth_forward", "lateral_camera_left", "vertical_camera_up"],
            "mujoco_camera_basis": ["x_image_right", "y_image_up", "negative_z_forward"],
        },
    }


def _configured_front_camera_randomization(
    camera: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    label = f"camera {camera.get('name')!r} randomization"
    _strict_object_keys(
        raw,
        label=label,
        required={
            "enabled",
            "mode",
            "azimuth",
            "table_edge_clearance_m",
            "height_offset_in_reference_frame_m",
            "look_at",
            "fovy_deg",
            "visibility_gate",
        },
    )
    if str((camera.get("mount") or {}).get("type") or "") != "world":
        raise ValueError(f"{label} mode {FRONT_CAMERA_RANDOMIZATION_MODE!r} requires a world camera.")
    extrinsics = camera.get("extrinsics") or {}
    if str(extrinsics.get("type") or "") != "body_frame_relative_lookat":
        raise ValueError(
            f"{label} mode {FRONT_CAMERA_RANDOMIZATION_MODE!r} requires "
            "body_frame_relative_lookat extrinsics."
        )
    azimuth_raw = _strict_object_keys(
        raw["azimuth"],
        label=f"{label}.azimuth",
        required={"bounds", "distribution"},
    )
    if str(azimuth_raw["bounds"]) != "reference_base_to_table_positive_x_edge_corners":
        raise ValueError(f"{label}.azimuth.bounds has an unsupported value.")
    if str(azimuth_raw["distribution"]) != "uniform_angle":
        raise ValueError(f"{label}.azimuth.distribution must be 'uniform_angle'.")
    clearance_raw = _strict_object_keys(
        raw["table_edge_clearance_m"],
        label=f"{label}.table_edge_clearance_m",
        required={"minimum", "maximum", "distribution"},
    )
    if str(clearance_raw["distribution"]) != "uniform":
        raise ValueError(f"{label}.table_edge_clearance_m.distribution must be 'uniform'.")
    clearance_min = _finite_float(
        clearance_raw["minimum"], label=f"{label}.table_edge_clearance_m.minimum", minimum=0.0
    )
    clearance_max = _finite_float(
        clearance_raw["maximum"], label=f"{label}.table_edge_clearance_m.maximum", minimum=clearance_min
    )
    height_raw = _strict_object_keys(
        raw["height_offset_in_reference_frame_m"],
        label=f"{label}.height_offset_in_reference_frame_m",
        required={"center", "half_range", "distribution"},
    )
    if str(height_raw["distribution"]) != "uniform_symmetric":
        raise ValueError(
            f"{label}.height_offset_in_reference_frame_m.distribution must be 'uniform_symmetric'."
        )
    height_center = _finite_float(
        height_raw["center"], label=f"{label}.height_offset_in_reference_frame_m.center"
    )
    height_half_range = _finite_float(
        height_raw["half_range"],
        label=f"{label}.height_offset_in_reference_frame_m.half_range",
        minimum=0.0,
    )
    position_offset = np.asarray(extrinsics.get("position_offset") or [], dtype=np.float64)
    if position_offset.shape != (3,) or not math.isclose(
        height_center, float(position_offset[2]), rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"{label} height center must equal extrinsics.position_offset.z={position_offset.tolist()}."
        )
    look_at_raw = _strict_object_keys(
        raw["look_at"],
        label=f"{label}.look_at",
        required={"center_offset_in_reference_frame_m", "disk_plane", "disk_diameter_m", "distribution"},
    )
    if str(look_at_raw["disk_plane"]) != "reference_xy":
        raise ValueError(f"{label}.look_at.disk_plane must be 'reference_xy'.")
    if str(look_at_raw["distribution"]) != "uniform_area":
        raise ValueError(f"{label}.look_at.distribution must be 'uniform_area'.")
    look_at_center = np.asarray(look_at_raw["center_offset_in_reference_frame_m"], dtype=np.float64)
    target_offset = np.asarray(extrinsics.get("target_offset") or [], dtype=np.float64)
    if look_at_center.shape != (3,) or not np.all(np.isfinite(look_at_center)):
        raise ValueError(f"{label}.look_at.center_offset_in_reference_frame_m must have three finite values.")
    if target_offset.shape != (3,) or not np.allclose(look_at_center, target_offset, atol=1e-9):
        raise ValueError(
            f"{label} look-at center must equal extrinsics.target_offset={target_offset.tolist()}."
        )
    disk_diameter = _finite_float(
        look_at_raw["disk_diameter_m"],
        label=f"{label}.look_at.disk_diameter_m",
        minimum=0.0,
    )
    visibility_raw = _strict_object_keys(
        raw["visibility_gate"],
        label=f"{label}.visibility_gate",
        required={
            "max_attempts",
            "workspace_center_offset_in_reference_frame_m",
            "horizontal_radius_m",
            "height_range_in_reference_frame_m",
            "azimuth_samples",
            "render_aspect_ratio",
        },
    )
    max_attempts = int(visibility_raw["max_attempts"])
    azimuth_samples = int(visibility_raw["azimuth_samples"])
    if max_attempts < 1:
        raise ValueError(f"{label}.visibility_gate.max_attempts must be >= 1.")
    if azimuth_samples < 4:
        raise ValueError(f"{label}.visibility_gate.azimuth_samples must be >= 4.")
    workspace_center = np.asarray(
        visibility_raw["workspace_center_offset_in_reference_frame_m"], dtype=np.float64
    )
    height_range = np.asarray(visibility_raw["height_range_in_reference_frame_m"], dtype=np.float64)
    aspect_ratio = np.asarray(visibility_raw["render_aspect_ratio"], dtype=np.float64)
    if workspace_center.shape != (3,) or not np.all(np.isfinite(workspace_center)):
        raise ValueError(f"{label}.visibility_gate workspace center must have three finite values.")
    if height_range.shape != (2,) or not np.all(np.isfinite(height_range)) or height_range[1] < height_range[0]:
        raise ValueError(f"{label}.visibility_gate height range must be [minimum, maximum].")
    if aspect_ratio.shape != (2,) or np.any(aspect_ratio <= 0.0):
        raise ValueError(f"{label}.visibility_gate.render_aspect_ratio must contain two positive values.")
    required_aspect = str((camera.get("intrinsics") or {}).get("required_aspect_ratio") or "").strip()
    required_match = re.fullmatch(r"([1-9][0-9]*):([1-9][0-9]*)", required_aspect)
    if required_match is None:
        raise ValueError(f"{label} camera intrinsics must declare required_aspect_ratio as W:H.")
    required_ratio = float(required_match.group(1)) / float(required_match.group(2))
    configured_gate_ratio = float(aspect_ratio[0]) / float(aspect_ratio[1])
    if not math.isclose(configured_gate_ratio, required_ratio, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{label}.visibility_gate.render_aspect_ratio must match intrinsics.required_aspect_ratio."
        )
    configured_fovy = _camera_fovy(camera)
    return {
        "enabled": bool(raw["enabled"]),
        "mode": FRONT_CAMERA_RANDOMIZATION_MODE,
        "azimuth": {
            "bounds": "reference_base_to_table_positive_x_edge_corners",
            "distribution": "uniform_angle",
        },
        "table_edge_clearance_m": {
            "minimum": clearance_min,
            "maximum": clearance_max,
            "distribution": "uniform",
        },
        "height_offset_in_reference_frame_m": {
            "center": height_center,
            "half_range": height_half_range,
            "distribution": "uniform_symmetric",
        },
        "look_at": {
            "center_offset_in_reference_frame_m": look_at_center.tolist(),
            "disk_plane": "reference_xy",
            "disk_diameter_m": disk_diameter,
            "distribution": "uniform_area",
        },
        "fovy_deg": _configured_fovy_randomization(
            raw["fovy_deg"],
            label=f"{label}.fovy_deg",
            configured_fovy=configured_fovy,
        ),
        "visibility_gate": {
            "max_attempts": max_attempts,
            "workspace_center_offset_in_reference_frame_m": workspace_center.tolist(),
            "horizontal_radius_m": _finite_float(
                visibility_raw["horizontal_radius_m"],
                label=f"{label}.visibility_gate.horizontal_radius_m",
                minimum=0.0,
            ),
            "height_range_in_reference_frame_m": height_range.tolist(),
            "azimuth_samples": azimuth_samples,
            "render_aspect_ratio": aspect_ratio.tolist(),
        },
    }


def _camera_randomization_by_name(camera_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for camera in camera_config["cameras"]:
        randomization = deepcopy(camera.get("randomization") or {})
        if not isinstance(randomization, dict):
            raise ValueError(f"camera {camera.get('name')!r} randomization must be an object.")
        mode = str(randomization.get("mode") or "").strip()
        if mode == WRIST_CAMERA_RANDOMIZATION_MODE:
            randomization = _configured_wrist_camera_randomization(camera, randomization)
        elif mode == FRONT_CAMERA_RANDOMIZATION_MODE:
            randomization = _configured_front_camera_randomization(camera, randomization)
        elif mode:
            raise ValueError(f"Unsupported camera randomization mode {mode!r}.")
        else:
            randomization.setdefault("enabled", False)
            randomization.setdefault("randomize_position", bool(randomization["enabled"]))
            randomization.setdefault("randomize_rotation", bool(randomization["enabled"]))
            randomization.setdefault("randomize_fovy", bool(randomization["enabled"]))
            randomization.setdefault("position_perturbation_size", 0.0)
            randomization.setdefault("rotation_perturbation_size", 0.0)
            randomization.setdefault("fovy_perturbation_size", 0.0)
        result[str(camera["name"])] = randomization
    return result


def _absolutize_xml_file_attrs(root: ET.Element, *, xml_dir: Path) -> dict[str, Any]:
    resolved: list[dict[str, str]] = []
    resolved_count = 0
    for elem in root.iter():
        if elem.tag not in XML_FILE_ATTR_TAGS:
            continue
        raw_file = elem.get("file")
        if not raw_file or raw_file.startswith(("http://", "https://", "s3://")):
            continue
        raw_path = Path(raw_file)
        if raw_path.is_absolute():
            continue
        resolved_path = (xml_dir / raw_path).resolve()
        elem.set("file", str(resolved_path))
        resolved_count += 1
        if len(resolved) < 40:
            resolved.append({"tag": elem.tag, "from": raw_file, "to": str(resolved_path)})
    return {"resolved_file_attr_count": resolved_count, "resolved_file_attr_examples": resolved}


def _parse_vec(raw: str | None, *, length: int, default: tuple[float, ...] | None = None) -> np.ndarray:
    if raw is None or not str(raw).strip():
        if default is None:
            return np.zeros(length, dtype=np.float64)
        return np.asarray(default, dtype=np.float64)
    values = [float(part) for part in str(raw).split()]
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {len(values)} in {raw!r}.")
    return np.asarray(values, dtype=np.float64)


def _set_body_pos(body: ET.Element, pos: np.ndarray) -> None:
    body.set("pos", _format_float_sequence(np.asarray(pos, dtype=np.float64)))


def _body_has_direct_free_joint(body: ET.Element) -> bool:
    for child in list(body):
        if child.tag == "freejoint":
            return True
        if child.tag == "joint" and child.get("type") == "free":
            return True
    return False


def _name_contains(name: str, needles: tuple[str, ...]) -> bool:
    lower = name.lower()
    return any(needle in lower for needle in needles)


def _is_object_free_body(body: ET.Element) -> bool:
    name = body.get("name") or ""
    if not _body_has_direct_free_joint(body):
        return False
    if _name_contains(name, ARM_ROBOT_BODY_NEEDLES + BOOSTER_BODY_NEEDLES):
        return False
    return _name_contains(name, OBJECT_BODY_NAME_NEEDLES)


def _direct_worldbody_bodies(root: ET.Element) -> list[ET.Element]:
    bodies: list[ET.Element] = []
    for worldbody in root.findall("worldbody"):
        bodies.extend([child for child in list(worldbody) if child.tag == "body"])
    return bodies


def _direct_object_body_positions(root: ET.Element) -> list[np.ndarray]:
    positions: list[np.ndarray] = []
    for body in _direct_worldbody_bodies(root):
        if _is_object_free_body(body):
            positions.append(_parse_vec(body.get("pos"), length=3, default=(0.0, 0.0, 0.0)))
    return positions


def _direct_arm_robot_body_positions(root: ET.Element) -> list[np.ndarray]:
    positions: list[np.ndarray] = []
    for body in _direct_worldbody_bodies(root):
        name = body.get("name") or ""
        if _name_contains(name, ARM_ROBOT_BODY_NEEDLES):
            positions.append(_parse_vec(body.get("pos"), length=3, default=(0.0, 0.0, 0.0)))
    return positions


def _raw_xml_mentions_booster(root: ET.Element) -> bool:
    for elem in root.iter():
        for value in elem.attrib.values():
            if _name_contains(str(value), BOOSTER_BODY_NEEDLES):
                return True
    return False


def _extract_original_table_metadata(root: ET.Element) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for body in _direct_worldbody_bodies(root):
        if not _is_original_scene_shell_body(body):
            continue
        body_pos = _parse_vec(body.get("pos"), length=3, default=(0.0, 0.0, 0.0))
        top_candidates: list[dict[str, Any]] = []
        for geom in body.findall("geom"):
            name = geom.get("name") or ""
            lower = name.lower()
            if "top" not in lower and "surface" not in lower:
                continue
            if geom.get("size") is None:
                continue
            geom_pos = _parse_vec(geom.get("pos"), length=3, default=(0.0, 0.0, 0.0))
            size = _parse_vec(geom.get("size"), length=3, default=(0.0, 0.0, 0.0))
            top_candidates.append(
                {
                    "geom_name": name,
                    "center": (body_pos + geom_pos).tolist(),
                    "top_z": float(body_pos[2] + geom_pos[2] + size[2]),
                    "full_size": [float(size[0] * 2.0), float(size[1] * 2.0), float(size[2] * 2.0)],
                }
            )
        if not top_candidates:
            continue
        top = max(top_candidates, key=lambda item: float(item["top_z"]))
        tables.append(
            {
                "body_name": body.get("name") or "",
                "body_pos": body_pos.tolist(),
                "top_geom": top,
            }
        )
    return tables


def _infer_table_center_xy(root: ET.Element, *, include_robot: bool) -> tuple[float, float]:
    samples = _direct_object_body_positions(root)
    if include_robot:
        samples.extend(_direct_arm_robot_body_positions(root))
    if not samples:
        return DEFAULT_TABLE_CENTER_XY
    stacked = np.stack(samples, axis=0)
    center = np.mean(stacked[:, :2], axis=0)
    return (float(center[0]), float(center[1]))


def _infer_table_full_size(root: ET.Element, *, min_size: tuple[float, float, float]) -> tuple[float, float, float]:
    positions = _direct_object_body_positions(root)
    if not positions:
        return min_size
    stacked = np.stack(positions, axis=0)
    span = np.max(stacked[:, :2], axis=0) - np.min(stacked[:, :2], axis=0)
    return (
        float(max(min_size[0], span[0] + TABLE_TOP_MARGIN * 2.0)),
        float(max(min_size[1], span[1] + TABLE_TOP_MARGIN * 2.0)),
        float(min_size[2]),
    )


def _direct_free_object_body_metadata(root: ET.Element) -> list[dict[str, Any]]:
    """Return every direct free body that is not an embodiment root.

    The historical table-size heuristic intentionally recognizes only a small
    set of object-name tokens.  Fixed-table containment must be stricter: a
    knife, cutting board, or any other free object is still required to start
    above the physical tabletop even when its name is not in that legacy list.
    """

    objects: list[dict[str, Any]] = []
    for body in _direct_worldbody_bodies(root):
        name = body.get("name") or ""
        if not _body_has_direct_free_joint(body):
            continue
        if _name_contains(name, ARM_ROBOT_BODY_NEEDLES + BOOSTER_BODY_NEEDLES):
            continue
        objects.append(
            {
                "body_name": name,
                "center": _parse_vec(
                    body.get("pos"), length=3, default=(0.0, 0.0, 0.0)
                ).tolist(),
            }
        )
    return objects


def _fixed_body_frame_table_placement(
    root: ET.Element,
    placement_profile: dict[str, Any],
    *,
    dynamic_table_center_xy: tuple[float, float],
    dynamic_required_table_full_size: tuple[float, float, float],
) -> tuple[tuple[float, float], tuple[float, float, float], dict[str, Any]]:
    if not isinstance(placement_profile, dict):
        raise ValueError("render_profile.scene.table_placement must be a JSON object.")
    expected_fields = {
        "mode",
        "reference_body",
        "center_offset_xy_in_reference_frame_m",
        "full_size_m",
        "object_center_policy",
    }
    unknown_fields = sorted(set(placement_profile) - expected_fields)
    missing_fields = sorted(expected_fields - set(placement_profile))
    if unknown_fields or missing_fields:
        raise ValueError(
            "render_profile.scene.table_placement fields must match the fixed-table contract: "
            f"missing={missing_fields}, unknown={unknown_fields}."
        )

    mode = str(placement_profile["mode"] or "").strip()
    if mode != "fixed_body_frame":
        raise ValueError(
            "render_profile.scene.table_placement.mode must be 'fixed_body_frame', "
            f"got {mode!r}."
        )
    object_center_policy = str(placement_profile["object_center_policy"] or "").strip()
    if object_center_policy != "require_inside_fixed_footprint":
        raise ValueError(
            "render_profile.scene.table_placement.object_center_policy must be "
            f"'require_inside_fixed_footprint', got {object_center_policy!r}."
        )
    reference_body = str(placement_profile["reference_body"] or "").strip()
    if not reference_body:
        raise ValueError("render_profile.scene.table_placement.reference_body must be non-empty.")

    center_offset = np.asarray(
        placement_profile["center_offset_xy_in_reference_frame_m"], dtype=np.float64
    )
    if center_offset.shape != (2,) or not np.all(np.isfinite(center_offset)):
        raise ValueError(
            "render_profile.scene.table_placement.center_offset_xy_in_reference_frame_m "
            "must contain two finite values."
        )
    full_size = np.asarray(placement_profile["full_size_m"], dtype=np.float64)
    if full_size.shape != (3,) or not np.all(np.isfinite(full_size)) or np.any(full_size <= 0.0):
        raise ValueError(
            "render_profile.scene.table_placement.full_size_m must contain three positive finite values."
        )

    documents = [_MJCFDocument(root=root, path=None)]
    reference_match = _find_body_or_none(documents, reference_body)
    if reference_match is None:
        raise ValueError(
            f"Fixed-table reference body {reference_body!r} was not found in the entry MJCF."
        )
    reference_element, reference_document = reference_match
    if reference_document.path is not None:
        raise AssertionError("Internal error: the fixed-table reference must belong to the entry MJCF.")
    reference_world_pos, reference_world_quat = _body_static_world_pose(
        reference_element, _parent_map(documents)
    )
    center_offset_world = _quat_rotate_wxyz(
        reference_world_quat,
        np.asarray([center_offset[0], center_offset[1], 0.0], dtype=np.float64),
    )
    table_center_xy_array = reference_world_pos[:2] + center_offset_world[:2]
    half_size_xy = full_size[:2] / 2.0
    footprint_min_xy = table_center_xy_array - half_size_xy
    footprint_max_xy = table_center_xy_array + half_size_xy

    object_metadata = _direct_free_object_body_metadata(root)
    validated_objects: list[dict[str, Any]] = []
    minimum_axis_clearance: np.ndarray | None = None
    tolerance = 1e-9
    for item in object_metadata:
        center = np.asarray(item["center"], dtype=np.float64)
        axis_clearance = half_size_xy - np.abs(center[:2] - table_center_xy_array)
        if np.any(axis_clearance < -tolerance):
            raise ValueError(
                "Free object center lies outside the configured fixed Franka table footprint: "
                f"body={item['body_name']!r}, center_xy={center[:2].tolist()}, "
                f"footprint_min_xy={footprint_min_xy.tolist()}, "
                f"footprint_max_xy={footprint_max_xy.tolist()}."
            )
        minimum_axis_clearance = (
            axis_clearance.copy()
            if minimum_axis_clearance is None
            else np.minimum(minimum_axis_clearance, axis_clearance)
        )
        validated_objects.append(
            {
                **item,
                "edge_clearance_xy_m": axis_clearance.tolist(),
                "minimum_edge_clearance_m": float(np.min(axis_clearance)),
            }
        )

    dynamic_required = np.asarray(dynamic_required_table_full_size, dtype=np.float64)
    dynamic_overflow = np.maximum(dynamic_required - full_size, 0.0)
    return (
        (float(table_center_xy_array[0]), float(table_center_xy_array[1])),
        (float(full_size[0]), float(full_size[1]), float(full_size[2])),
        {
            "fixed_table_placement_applied": True,
            "fixed_table_placement": {
                "mode": mode,
                "reference_body": reference_body,
                "resolved_reference_body": reference_element.get("name") or reference_body,
                "reference_body_world_pos": reference_world_pos.tolist(),
                "reference_body_world_quat_wxyz": reference_world_quat.tolist(),
                "center_offset_xy_in_reference_frame_m": center_offset.tolist(),
                "resolved_table_center_xy_world_m": table_center_xy_array.tolist(),
                "full_size_m": full_size.tolist(),
                "footprint_min_xy_world_m": footprint_min_xy.tolist(),
                "footprint_max_xy_world_m": footprint_max_xy.tolist(),
                "object_center_policy": object_center_policy,
                "validated_object_count": len(validated_objects),
                "validated_objects": validated_objects,
                "minimum_edge_clearance_xy_m": (
                    minimum_axis_clearance.tolist()
                    if minimum_axis_clearance is not None
                    else None
                ),
                "minimum_edge_clearance_m": (
                    float(np.min(minimum_axis_clearance))
                    if minimum_axis_clearance is not None
                    else None
                ),
                "legacy_dynamic_table_center_xy_world_m": list(dynamic_table_center_xy),
                "legacy_dynamic_required_table_full_size_m": dynamic_required.tolist(),
                "legacy_dynamic_required_size_overflow_m": dynamic_overflow.tolist(),
                "legacy_aesthetic_margin_is_not_a_containment_requirement": True,
            },
        },
    )


def _infer_scene_placement(root: ET.Element, render_profile: dict[str, Any] | None = None) -> ScenePlacement:
    render_profile = render_profile or {}
    if not isinstance(render_profile, dict):
        raise ValueError("render_profile must be a JSON object.")
    scene_profile = render_profile.get("scene") or {}
    if not isinstance(scene_profile, dict):
        raise ValueError("render_profile.scene must be a JSON object.")
    original_tables = _extract_original_table_metadata(root)
    is_booster = bool(original_tables and _raw_xml_mentions_booster(root))
    table_placement_metadata: dict[str, Any] = {"fixed_table_placement_applied": False}
    if is_booster:
        table = original_tables[0]["top_geom"]
        center = table["center"]
        full_size = table["full_size"]
        preserve_source_table = bool(scene_profile.get("preserve_source_table", False))
        if preserve_source_table:
            table_full_size = tuple(float(value) for value in full_size)
            policy = "booster_source_table_preserved"
            table_surface_geom_names = (str(table["geom_name"]),)
        else:
            table_full_size = (
                float(max(DEFAULT_TABLE_FULL_SIZE[0], full_size[0])),
                float(max(DEFAULT_TABLE_FULL_SIZE[1], full_size[1])),
                float(DEFAULT_TABLE_FULL_SIZE[2]),
            )
            policy = "booster_object_table_robot_on_floor"
            table_surface_geom_names = (f"{ARENA_PREFIX}table_visual",)
        table_center_xy = (float(center[0]), float(center[1]))
        table_top_z = float(table["top_z"])
        arm_scene_z_offset = 0.0
        free_joint_qpos_z_offset = 0.0
    else:
        preserve_source_table = False
        policy = "arm_robot_and_objects_on_table"
        dynamic_table_center_xy = _infer_table_center_xy(root, include_robot=True)
        dynamic_required_table_full_size = _infer_table_full_size(
            root, min_size=DEFAULT_TABLE_FULL_SIZE
        )
        table_placement_profile = scene_profile.get("table_placement")
        if table_placement_profile is None:
            table_center_xy = dynamic_table_center_xy
            table_full_size = dynamic_required_table_full_size
            table_placement_metadata.update(
                {
                    "legacy_dynamic_table_center_xy_world_m": list(dynamic_table_center_xy),
                    "legacy_dynamic_required_table_full_size_m": list(
                        dynamic_required_table_full_size
                    ),
                }
            )
        else:
            (
                table_center_xy,
                table_full_size,
                table_placement_metadata,
            ) = _fixed_body_frame_table_placement(
                root,
                table_placement_profile,
                dynamic_table_center_xy=dynamic_table_center_xy,
                dynamic_required_table_full_size=dynamic_required_table_full_size,
            )
        table_top_z = DEFAULT_TABLE_TOP_Z
        arm_scene_z_offset = DEFAULT_TABLE_TOP_Z
        free_joint_qpos_z_offset = DEFAULT_TABLE_TOP_Z
        table_surface_geom_names = (f"{ARENA_PREFIX}table_visual",)
    metadata = {
        "scene_placement_policy": policy,
        "table_center_xy": list(table_center_xy),
        "table_top_z": float(table_top_z),
        "table_full_size": list(table_full_size),
        "preserve_source_table": bool(preserve_source_table),
        "table_geometry_policy": scene_profile.get("table_geometry_policy"),
        "arm_scene_z_offset": float(arm_scene_z_offset),
        "free_joint_qpos_z_offset": float(free_joint_qpos_z_offset),
        "table_surface_geom_names": list(table_surface_geom_names),
        "original_table_metadata": original_tables,
        "object_body_positions_for_table": [pos.tolist() for pos in _direct_object_body_positions(root)],
        "arm_robot_body_positions_for_table": [pos.tolist() for pos in _direct_arm_robot_body_positions(root)],
        "booster_detected": bool(is_booster),
        **table_placement_metadata,
    }
    return ScenePlacement(
        policy=policy,
        table_center_xy=table_center_xy,
        table_top_z=table_top_z,
        table_full_size=table_full_size,
        preserve_source_table=preserve_source_table,
        arm_scene_z_offset=arm_scene_z_offset,
        free_joint_qpos_z_offset=free_joint_qpos_z_offset,
        table_surface_geom_names=table_surface_geom_names,
        metadata=metadata,
    )


def _apply_arm_scene_z_offset(root: ET.Element, placement: ScenePlacement) -> dict[str, Any]:
    if abs(float(placement.arm_scene_z_offset)) <= 1e-12:
        return {"scene_body_z_offset_applied": False, "scene_body_z_offset": 0.0, "z_shifted_body_names": []}
    shifted: list[dict[str, Any]] = []
    for body in _direct_worldbody_bodies(root):
        name = body.get("name") or ""
        if _is_original_scene_shell_body(body):
            continue
        if not (_is_object_free_body(body) or _name_contains(name, ARM_ROBOT_BODY_NEEDLES)):
            continue
        old_pos = _parse_vec(body.get("pos"), length=3, default=(0.0, 0.0, 0.0))
        new_pos = old_pos.copy()
        new_pos[2] += float(placement.arm_scene_z_offset)
        _set_body_pos(body, new_pos)
        shifted.append({"body_name": name, "old_pos": old_pos.tolist(), "new_pos": new_pos.tolist()})
    return {
        "scene_body_z_offset_applied": True,
        "scene_body_z_offset": float(placement.arm_scene_z_offset),
        "z_shifted_body_names": [item["body_name"] for item in shifted],
        "z_shifted_body_examples": shifted[:40],
    }


def _direct_scene_element_name(elem: ET.Element, fallback_prefix: str, index: int) -> str:
    return elem.get("name") or f"{fallback_prefix}_{index}"


def _body_has_joint(body: ET.Element) -> bool:
    return any(descendant.tag == "joint" for descendant in body.iter())


def _body_subtree_names(body: ET.Element) -> list[str]:
    return [descendant.get("name") or "" for descendant in body.iter("body") if descendant.get("name")]


def _is_original_scene_shell_body(body: ET.Element) -> bool:
    name = (body.get("name") or "").lower()
    return "table" in name and not _body_has_joint(body)


def _remove_original_direct_scene_shell(
    root: ET.Element,
    *,
    preserve_table_bodies: bool = False,
) -> dict[str, Any]:
    removed: list[dict[str, str]] = []
    removed_body_names: list[str] = []
    preserved_table_body_names: list[str] = []
    for worldbody_index, worldbody in enumerate(root.findall("worldbody")):
        for child_index, child in enumerate(list(worldbody)):
            if child.tag == "body" and _is_original_scene_shell_body(child):
                body_names = _body_subtree_names(child)
                if preserve_table_bodies:
                    preserved_table_body_names.extend(body_names)
                    continue
                removed_body_names.extend(body_names)
                worldbody.remove(child)
                removed.append(
                    {
                        "worldbody_index": str(worldbody_index),
                        "tag": child.tag,
                        "name": _direct_scene_element_name(child, f"unnamed_{child.tag}", child_index),
                        "body_names_in_subtree": ",".join(body_names),
                    }
                )
                continue
            if child.tag not in {"geom", "camera", "light"}:
                continue
            worldbody.remove(child)
            removed.append(
                {
                    "worldbody_index": str(worldbody_index),
                    "tag": child.tag,
                    "name": _direct_scene_element_name(child, f"unnamed_{child.tag}", child_index),
                }
            )
    return {
        "removed_original_scene_shell": removed,
        "removed_original_scene_shell_body_names": sorted(dict.fromkeys(removed_body_names)),
        "preserved_source_table_body_names": sorted(dict.fromkeys(preserved_table_body_names)),
    }


def _remove_arena_table_body(arena_root: ET.Element) -> dict[str, Any]:
    removed: list[str] = []
    worldbody = arena_root.find("worldbody")
    if worldbody is not None:
        for child in list(worldbody):
            if child.tag == "body" and (child.get("name") or "").lower() == "table":
                removed.append(child.get("name") or "")
                worldbody.remove(child)
    if len(removed) != 1:
        raise ValueError(f"Expected one Robosuite TableArena table body, removed {removed}.")
    return {"removed_arena_table_body_names": removed}


def _rename_arena_tree(arena_root: ET.Element, requested_camera_names: tuple[str, ...]) -> dict[str, Any]:
    requested = set(requested_camera_names)
    camera_names_kept: list[str] = []
    name_map: dict[str, str] = {}
    unnamed_counters: dict[str, int] = {}

    for elem in arena_root.iter():
        old_name = elem.get("name")
        if not old_name and elem.tag == "light":
            count = unnamed_counters.get("light", 0)
            old_name = f"arena_light_{count}"
            unnamed_counters["light"] = count + 1
            elem.set("name", old_name)
        if not old_name:
            continue
        if elem.tag == "camera" and old_name in requested:
            camera_names_kept.append(old_name)
            continue
        new_name = f"{ARENA_PREFIX}{old_name}"
        elem.set("name", new_name)
        name_map[old_name] = new_name

    reference_attrs = (
        "material",
        "texture",
        "mesh",
        "class",
        "childclass",
        "joint",
        "body",
        "site",
        "target",
        "objname",
    )
    for elem in arena_root.iter():
        for attr in reference_attrs:
            value = elem.get(attr)
            if value in name_map:
                elem.set(attr, name_map[value])

    return {
        "arena_prefix": ARENA_PREFIX,
        "arena_renamed_count": len(name_map),
        "arena_camera_names_kept": sorted(camera_names_kept),
        "arena_renamed_examples": [
            {"from": old_name, "to": new_name} for old_name, new_name in list(name_map.items())[:40]
        ],
    }


def _remove_arena_mocap_targets(arena_root: ET.Element) -> dict[str, Any]:
    removed: list[str] = []
    for parent in arena_root.iter():
        for child in list(parent):
            if child.tag == "body" and child.get("mocap") == "true":
                removed.append(child.get("name") or "")
                parent.remove(child)
    return {"removed_arena_mocap_bodies": removed}


def _append_children(dst_parent: ET.Element, src_parent: ET.Element | None) -> int:
    if src_parent is None:
        return 0
    count = 0
    for child in list(src_parent):
        dst_parent.append(deepcopy(child))
        count += 1
    return count


def _ensure_child(root: ET.Element, tag: str) -> ET.Element:
    child = root.find(tag)
    if child is None:
        child = ET.SubElement(root, tag)
    return child


def _merge_robosuite_table_arena(
    root: ET.Element,
    camera_names: tuple[str, ...],
    placement: ScenePlacement,
) -> dict[str, Any]:
    table_offset = (float(placement.table_center_xy[0]), float(placement.table_center_xy[1]), float(placement.table_top_z))
    arena = TableArena(table_full_size=placement.table_full_size, table_offset=table_offset, has_legs=True)
    arena_root = ET.fromstring(arena.get_xml())
    arena_table_metadata = (
        _remove_arena_table_body(arena_root)
        if placement.preserve_source_table
        else {"removed_arena_table_body_names": []}
    )
    rename_metadata = _rename_arena_tree(arena_root, camera_names)
    mocap_metadata = _remove_arena_mocap_targets(arena_root)
    original_removal_metadata = _remove_original_direct_scene_shell(
        root,
        preserve_table_bodies=placement.preserve_source_table,
    )

    asset = _ensure_child(root, "asset")
    worldbody = _ensure_child(root, "worldbody")
    arena_asset_count = _append_children(asset, arena_root.find("asset"))
    arena_worldbody_count = _append_children(worldbody, arena_root.find("worldbody"))

    return {
        "arena_backend": "robosuite.models.arenas.TableArena",
        "arena_table_full_size": list(placement.table_full_size),
        "arena_table_offset": list(table_offset),
        "arena_table_surface_z": float(placement.table_top_z),
        "arena_has_table": not placement.preserve_source_table,
        "arena_has_legs": not placement.preserve_source_table,
        "source_table_preserved": bool(placement.preserve_source_table),
        "arena_asset_child_count": arena_asset_count,
        "arena_worldbody_child_count": arena_worldbody_count,
        **arena_table_metadata,
        **rename_metadata,
        **mocap_metadata,
        **original_removal_metadata,
        **placement.metadata,
    }


def _profile_vector(
    value: Any,
    *,
    length: int,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{field_name} must contain {length} finite values.")
    if minimum is not None and np.any(result < minimum):
        raise ValueError(f"{field_name} values must be >= {minimum}.")
    if maximum is not None and np.any(result > maximum):
        raise ValueError(f"{field_name} values must be <= {maximum}.")
    return result


def _configure_reference_lighting_rig(
    root: ET.Element,
    camera_config: dict[str, Any],
    placement: ScenePlacement,
) -> dict[str, Any]:
    render_profile = camera_config.get("render_profile") or {}
    lighting_profile = render_profile.get("lighting") or {}
    if not lighting_profile:
        return {"lighting_rig_configured": False}
    if not isinstance(lighting_profile, dict):
        raise ValueError("render_profile.lighting must be a JSON object.")
    mode = str(lighting_profile.get("mode") or "").strip()
    if mode != "reference_five_light_rig":
        return {"lighting_rig_configured": False, "lighting_rig_mode": mode}
    sources = lighting_profile.get("sources") or []
    if not isinstance(sources, list) or len(sources) != 5:
        raise ValueError("reference_five_light_rig requires exactly five light sources.")

    worldbody = _ensure_child(root, "worldbody")
    removed_light_names: list[str] = []
    for child in list(worldbody):
        if child.tag == "light" and str(child.get("name") or "").startswith(ARENA_PREFIX):
            removed_light_names.append(str(child.get("name") or ""))
            worldbody.remove(child)

    table_origin = np.asarray(
        [placement.table_center_xy[0], placement.table_center_xy[1], placement.table_top_z],
        dtype=np.float64,
    )
    source_metadata: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"lighting source {index} must be a JSON object.")
        source_name = str(source.get("name") or "").strip()
        if not source_name or source_name in used_names:
            raise ValueError(f"lighting source {index} requires a unique non-empty name.")
        used_names.add(source_name)
        light_name = f"{ARENA_PREFIX}{source_name}"
        if "position" in source:
            position = _profile_vector(
                source["position"], length=3, field_name=f"lighting.sources[{index}].position"
            )
            position_frame = "world"
        else:
            position_offset = _profile_vector(
                source.get("position_offset") or [],
                length=3,
                field_name=f"lighting.sources[{index}].position_offset",
            )
            position = table_origin + position_offset
            position_frame = "table_surface"
        if "direction" in source:
            direction = _profile_vector(
                source["direction"], length=3, field_name=f"lighting.sources[{index}].direction"
            )
        else:
            target_offset = _profile_vector(
                source.get("target_offset") or [0.0, 0.0, 0.0],
                length=3,
                field_name=f"lighting.sources[{index}].target_offset",
            )
            direction = table_origin + target_offset - position
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-8:
            raise ValueError(f"lighting source {source_name!r} direction must be non-zero.")
        direction /= direction_norm

        light_attrs = {
            "name": light_name,
            "mode": "fixed",
            "active": "true",
            "pos": _format_float_sequence(position),
            "dir": _format_float_sequence(direction),
            "directional": "true" if bool(source.get("directional", False)) else "false",
            "castshadow": "true" if bool(source.get("castshadow", False)) else "false",
        }
        resolved_values: dict[str, Any] = {}
        for field_name in ("ambient", "diffuse", "specular"):
            values = _profile_vector(
                source.get(field_name) or [],
                length=3,
                field_name=f"lighting.sources[{index}].{field_name}",
                minimum=0.0,
                maximum=1.0,
            )
            light_attrs[field_name] = _format_float_sequence(values)
            resolved_values[field_name] = values.tolist()
        attenuation = _profile_vector(
            source.get("attenuation") or [1.0, 0.0, 0.0],
            length=3,
            field_name=f"lighting.sources[{index}].attenuation",
            minimum=0.0,
        )
        cutoff = float(source.get("cutoff", 90.0))
        exponent = float(source.get("exponent", 1.0))
        if not math.isfinite(cutoff) or not 0.0 < cutoff <= 90.0:
            raise ValueError(f"lighting.sources[{index}].cutoff must be in (0, 90].")
        if not math.isfinite(exponent) or exponent < 0.0:
            raise ValueError(f"lighting.sources[{index}].exponent must be >= 0.")
        light_attrs.update(
            {
                "attenuation": _format_float_sequence(attenuation),
                "cutoff": f"{cutoff:.9g}",
                "exponent": f"{exponent:.9g}",
            }
        )
        ET.SubElement(worldbody, "light", light_attrs)
        source_metadata.append(
            {
                "name": light_name,
                "source_type": source.get("source_type"),
                "position": position.tolist(),
                "position_frame": position_frame,
                "direction": direction.tolist(),
                "directional": bool(source.get("directional", False)),
                "castshadow": bool(source.get("castshadow", False)),
                "attenuation": attenuation.tolist(),
                "cutoff": cutoff,
                "exponent": exponent,
                **resolved_values,
            }
        )

    quality_profile = lighting_profile.get("render_quality") or {}
    if not isinstance(quality_profile, dict):
        raise ValueError("render_profile.lighting.render_quality must be a JSON object.")
    shadowsize = int(quality_profile.get("shadowsize", 4096))
    offsamples = int(quality_profile.get("offsamples", 4))
    shadowclip = float(quality_profile.get("shadowclip", 2.8))
    shadowscale = float(quality_profile.get("shadowscale", 0.8))
    if shadowsize < 128 or offsamples < 1:
        raise ValueError("lighting render quality requires shadowsize >= 128 and offsamples >= 1.")
    if not math.isfinite(shadowclip) or shadowclip <= 0.0:
        raise ValueError("lighting render quality requires shadowclip > 0.")
    if not math.isfinite(shadowscale) or shadowscale <= 0.0:
        raise ValueError("lighting render quality requires shadowscale > 0.")
    visual = _ensure_child(root, "visual")
    quality = visual.find("quality")
    if quality is None:
        quality = ET.SubElement(visual, "quality")
    quality.set("shadowsize", str(shadowsize))
    quality.set("offsamples", str(offsamples))
    visual_map = visual.find("map")
    if visual_map is None:
        visual_map = ET.SubElement(visual, "map")
    visual_map.set("shadowclip", f"{shadowclip:.9g}")
    visual_map.set("shadowscale", f"{shadowscale:.9g}")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    disable_headlight = bool(quality_profile.get("disable_headlight", True))
    headlight_values: dict[str, list[float]] = {}
    if disable_headlight:
        for field_name in ("ambient", "diffuse", "specular"):
            headlight.set(field_name, "0 0 0")
            headlight_values[field_name] = [0.0, 0.0, 0.0]
    else:
        for field_name in ("ambient", "diffuse", "specular"):
            values = _profile_vector(
                quality_profile.get(f"headlight_{field_name}") or [],
                length=3,
                field_name=f"render_profile.lighting.render_quality.headlight_{field_name}",
                minimum=0.0,
                maximum=1.0,
            )
            headlight.set(field_name, _format_float_sequence(values))
            headlight_values[field_name] = values.tolist()

    return {
        "lighting_rig_configured": True,
        "lighting_rig_mode": mode,
        "lighting_rig_source_style": lighting_profile.get("source_style"),
        "lighting_rig_removed_arena_lights": removed_light_names,
        "lighting_rig_sources": source_metadata,
        "lighting_render_quality": {
            "shadowsize": shadowsize,
            "offsamples": offsamples,
            "shadowclip": shadowclip,
            "shadowscale": shadowscale,
            "headlight_active": not disable_headlight,
            "headlight": headlight_values,
        },
    }


def _load_mjcf_documents(root: ET.Element, *, entry_xml: Path) -> list[_MJCFDocument]:
    """Load the entry document plus local MJCF includes so mounted cameras can target included bodies."""

    entry_xml = Path(entry_xml).expanduser().resolve()
    documents = [_MJCFDocument(root=root, path=None)]
    seen_paths = {entry_xml}

    def visit(document_root: ET.Element, xml_dir: Path) -> None:
        for include in document_root.iter("include"):
            raw_file = str(include.get("file") or "").strip()
            if not raw_file or raw_file.startswith(("http://", "https://", "s3://")):
                continue
            include_path = Path(raw_file).expanduser()
            if not include_path.is_absolute():
                include_path = (xml_dir / include_path).resolve()
            else:
                include_path = include_path.resolve()
            if include_path in seen_paths:
                continue
            if not include_path.is_file():
                raise FileNotFoundError(
                    f"MJCF include required for mounted-camera resolution does not exist: {include_path}"
                )
            seen_paths.add(include_path)
            include_root = ET.parse(include_path).getroot()
            documents.append(_MJCFDocument(root=include_root, path=include_path))
            visit(include_root, include_path.parent)

    visit(root, entry_xml.parent)
    return documents


def _atomic_write_xml(root: ET.Element, path: Path) -> None:
    path = Path(path).expanduser().resolve()
    temp_path = path.with_name(f".{path.name}.tmp_{os.getpid()}_{time.time_ns()}")
    try:
        ET.ElementTree(root).write(temp_path, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _parent_map(documents: list[_MJCFDocument]) -> dict[ET.Element, ET.Element]:
    return {
        child: parent
        for document in documents
        for parent in document.root.iter()
        for child in list(parent)
    }


def _find_body_or_none(
    documents: list[_MJCFDocument],
    body_name: str,
) -> tuple[ET.Element, _MJCFDocument] | None:
    exact_matches: list[tuple[ET.Element, _MJCFDocument]] = []
    suffix_matches: list[tuple[ET.Element, _MJCFDocument]] = []
    for document in documents:
        for body in document.root.iter("body"):
            name = body.get("name") or ""
            if name == body_name:
                exact_matches.append((body, document))
            if name.endswith(f"/{body_name}"):
                suffix_matches.append((body, document))
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError(f"Camera body {body_name!r} is duplicated in the final MJCF.")
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        match_names = [body.get("name") or "" for body, _ in suffix_matches]
        raise ValueError(f"Camera mount body {body_name!r} matched multiple prefixed bodies: {match_names}")
    return None


def _find_body(documents: list[_MJCFDocument], body_name: str) -> tuple[ET.Element, _MJCFDocument]:
    match = _find_body_or_none(documents, body_name)
    if match is not None:
        return match
    raise ValueError(f"Camera mount body {body_name!r} was not found in the final MJCF or its includes.")


def _body_name_matches(resolved_name: str, requested_name: str) -> bool:
    return resolved_name == requested_name or resolved_name.endswith(f"/{requested_name}")


def _body_static_world_pose(
    body: ET.Element,
    parents: dict[ET.Element, ET.Element],
) -> tuple[np.ndarray, np.ndarray]:
    chain: list[ET.Element] = []
    cursor: ET.Element | None = body
    while cursor is not None and cursor.tag == "body":
        chain.append(cursor)
        parent = parents.get(cursor)
        cursor = parent if parent is not None and parent.tag == "body" else None

    world_pos = np.zeros(3, dtype=np.float64)
    world_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for chain_body in reversed(chain):
        body_name = chain_body.get("name") or "<unnamed>"
        if chain_body.find("freejoint") is not None or chain_body.find("joint") is not None:
            raise ValueError(
                f"Camera reference body {body.get('name')!r} has dynamic ancestor {body_name!r}; "
                "a fixed base frame is required."
            )
        unsupported_orientation_attrs = [
            attr
            for attr in ("axisangle", "euler", "xyaxes", "zaxis")
            if chain_body.get(attr) is not None
        ]
        if unsupported_orientation_attrs:
            raise ValueError(
                f"Camera reference body {body_name!r} uses unsupported static orientation "
                f"attributes {unsupported_orientation_attrs}; use an explicit MJCF quat."
            )
        local_pos = _parse_vec(chain_body.get("pos"), length=3, default=(0.0, 0.0, 0.0))
        local_quat = _normalize_quat_wxyz(
            _parse_vec(chain_body.get("quat"), length=4, default=(1.0, 0.0, 0.0, 0.0)),
            label=f"body {body_name!r} quat",
        )
        world_pos = world_pos + _quat_rotate_wxyz(world_quat, local_pos)
        world_quat = _normalize_quat_wxyz(
            _quat_multiply_wxyz(world_quat, local_quat),
            label=f"body {body_name!r} world quat",
        )
    return world_pos, world_quat


def _body_frame_relative_camera_pose(
    camera_config: dict[str, Any],
    documents: list[_MJCFDocument],
    parents: dict[ET.Element, ET.Element],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    extrinsics = camera_config.get("extrinsics") or {}
    reference_body = str(extrinsics.get("reference_body") or "").strip()
    if not reference_body:
        raise ValueError(
            f"Body-frame-relative camera {camera_config.get('name')!r} must define "
            "extrinsics.reference_body."
        )
    for field_name in ("position_offset", "target_offset"):
        if field_name not in extrinsics:
            raise ValueError(
                f"Body-frame-relative camera {camera_config.get('name')!r} must define "
                f"extrinsics.{field_name}."
            )
    body, body_document = _find_body(documents, reference_body)
    resolved_reference_body = body.get("name") or reference_body
    if body_document.path is not None:
        raise ValueError(
            f"Camera reference body {resolved_reference_body!r} is defined in included MJCF "
            f"{body_document.path}; body_frame_relative_lookat currently requires a fixed body "
            "in the entry MJCF so no include-parent transform can be omitted."
        )
    base_pos, base_quat = _body_static_world_pose(body, parents)
    position_offset = _parse_vec(
        _format_float_sequence(extrinsics.get("position_offset") or []),
        length=3,
    )
    target_offset = _parse_vec(
        _format_float_sequence(extrinsics.get("target_offset") or []),
        length=3,
    )
    position = base_pos + _quat_rotate_wxyz(base_quat, position_offset)
    target = base_pos + _quat_rotate_wxyz(base_quat, target_offset)
    return position, target, {
        "extrinsics_type": "body_frame_relative_lookat",
        "reference_body": reference_body,
        "resolved_reference_body": resolved_reference_body,
        "reference_document": "entry_mjcf",
        "reference_body_world_pos": base_pos.tolist(),
        "reference_body_world_quat_wxyz": base_quat.tolist(),
        "position_offset_in_reference_body": position_offset.tolist(),
        "target_offset_in_reference_body": target_offset.tolist(),
    }


def _resolve_camera_mount_body(
    documents: list[_MJCFDocument],
    mount: dict[str, Any],
    parents: dict[ET.Element, ET.Element],
    element_documents: dict[ET.Element, _MJCFDocument],
) -> tuple[ET.Element, _MJCFDocument, dict[str, Any]]:
    mount_body = str(mount.get("body") or "").strip()
    if not mount_body:
        raise ValueError("Body-mounted camera must define mount.body.")
    existing = _find_body_or_none(documents, mount_body)
    if existing is not None:
        body, document = existing
        return body, document, {
            "type": "existing_body",
            "requested_body": mount_body,
            "resolved_body": body.get("name") or mount_body,
            "injected": False,
        }

    adapter = mount.get("missing_body_frame_adapter")
    if not isinstance(adapter, dict):
        raise ValueError(
            f"Camera mount body {mount_body!r} is missing and no explicit "
            "mount.missing_body_frame_adapter was configured."
        )
    adapter_type = str(adapter.get("type") or "").strip()
    if adapter_type != "validated_identity_child":
        raise ValueError(f"Unsupported missing body frame adapter {adapter_type!r}.")
    source_body_name = str(adapter.get("source_body") or "").strip()
    if not source_body_name:
        raise ValueError("validated_identity_child requires source_body.")
    source_body, source_document = _find_body(documents, source_body_name)
    resolved_source_body = source_body.get("name") or source_body_name
    source_parent = parents.get(source_body)
    expected_source_parent = str(adapter.get("expected_source_parent") or "").strip()
    if not expected_source_parent:
        raise ValueError("validated_identity_child requires expected_source_parent.")
    resolved_source_parent = (
        source_parent.get("name")
        if source_parent is not None and source_parent.tag == "body"
        else None
    )
    if resolved_source_parent is None or not _body_name_matches(
        resolved_source_parent, expected_source_parent
    ):
        raise ValueError(
            f"Camera frame adapter expected source body {resolved_source_body!r} under "
            f"{expected_source_parent!r}, got {resolved_source_parent!r}."
        )

    unsupported_orientation_attrs = [
        attr
        for attr in ("axisangle", "euler", "xyaxes", "zaxis")
        if source_body.get(attr) is not None
    ]
    if unsupported_orientation_attrs:
        raise ValueError(
            f"Camera frame adapter source body {resolved_source_body!r} uses unsupported "
            f"orientation attributes {unsupported_orientation_attrs}."
        )
    for field_name in ("expected_source_local_pos", "expected_source_local_quat_wxyz"):
        if field_name not in adapter:
            raise ValueError(f"validated_identity_child requires {field_name}.")
    actual_source_pos = _parse_vec(
        source_body.get("pos"), length=3, default=(0.0, 0.0, 0.0)
    )
    actual_source_quat = _normalize_quat_wxyz(
        _parse_vec(
            source_body.get("quat"),
            length=4,
            default=(1.0, 0.0, 0.0, 0.0),
        ),
        label=f"body {resolved_source_body!r} quat",
    )
    expected_source_pos = _parse_vec(
        _format_float_sequence(adapter["expected_source_local_pos"]),
        length=3,
    )
    expected_source_quat = _normalize_quat_wxyz(
        _parse_vec(
            _format_float_sequence(adapter["expected_source_local_quat_wxyz"]),
            length=4,
        ),
        label="expected source body quat",
    )
    tolerance = float(adapter.get("validation_tolerance", 1e-6))
    if tolerance <= 0.0:
        raise ValueError("Camera frame adapter validation_tolerance must be positive.")
    if not np.allclose(actual_source_pos, expected_source_pos, atol=tolerance, rtol=0.0):
        raise ValueError(
            f"Camera frame adapter source position mismatch for {resolved_source_body!r}: "
            f"expected={expected_source_pos.tolist()}, actual={actual_source_pos.tolist()}."
        )
    quat_matches = np.allclose(
        actual_source_quat, expected_source_quat, atol=tolerance, rtol=0.0
    ) or np.allclose(actual_source_quat, -expected_source_quat, atol=tolerance, rtol=0.0)
    if not quat_matches:
        raise ValueError(
            f"Camera frame adapter source quaternion mismatch for {resolved_source_body!r}: "
            f"expected={expected_source_quat.tolist()}, actual={actual_source_quat.tolist()}."
        )

    if "/" in mount_body:
        resolved_mount_body = mount_body
    elif "/" in resolved_source_body:
        resolved_mount_body = f"{resolved_source_body.rsplit('/', 1)[0]}/{mount_body}"
    else:
        resolved_mount_body = mount_body
    target_body = ET.SubElement(
        source_body,
        "body",
        {
            "name": resolved_mount_body,
            "pos": "0 0 0",
            "quat": "1 0 0 0",
        },
    )
    parents[target_body] = source_body
    element_documents[target_body] = source_document
    return target_body, source_document, {
        "type": adapter_type,
        "requested_body": mount_body,
        "resolved_body": resolved_mount_body,
        "source_body": source_body_name,
        "resolved_source_body": resolved_source_body,
        "expected_source_parent": expected_source_parent,
        "resolved_source_parent": resolved_source_parent,
        "source_local_pos": actual_source_pos.tolist(),
        "source_local_quat_wxyz": actual_source_quat.tolist(),
        "validation_tolerance": tolerance,
        "injected": True,
        "alias_local_pos": [0.0, 0.0, 0.0],
        "alias_local_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }


def _write_modified_include_documents(
    documents: list[_MJCFDocument],
    modified_include_paths: set[Path],
) -> list[str]:
    written_paths: list[str] = []
    for document in documents:
        if document.path is None or document.path not in modified_include_paths:
            continue
        _atomic_write_xml(document.root, document.path)
        written_paths.append(str(document.path))
    return written_paths


def _ensure_configured_cameras(
    root: ET.Element,
    camera_config: dict[str, Any],
    placement: ScenePlacement,
    *,
    entry_xml: Path,
) -> dict[str, Any]:
    requires_mjcf_documents = any(
        str((camera.get("mount") or {}).get("type") or "world") == "body"
        or str((camera.get("extrinsics") or {}).get("type") or "")
        == "body_frame_relative_lookat"
        for camera in camera_config["cameras"]
    )
    documents = (
        _load_mjcf_documents(root, entry_xml=entry_xml)
        if requires_mjcf_documents
        else [_MJCFDocument(root=root, path=None)]
    )
    element_documents = {
        element: document
        for document in documents
        for element in document.root.iter()
    }
    parents = _parent_map(documents)
    modified_include_paths: set[Path] = set()

    def mark_document_modified(document: _MJCFDocument | None) -> None:
        if document is not None and document.path is not None:
            modified_include_paths.add(document.path)

    resolved_mjcf_include_file_attr_count = 0
    for document in documents:
        if document.path is None:
            continue
        include_path_metadata = _absolutize_xml_file_attrs(document.root, xml_dir=document.path.parent)
        resolved_count = int(include_path_metadata["resolved_file_attr_count"])
        if resolved_count:
            resolved_mjcf_include_file_attr_count += resolved_count
            mark_document_modified(document)

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    existing_by_name = {
        elem.get("name"): elem
        for document in documents
        for elem in document.root.iter("camera")
        if elem.get("name")
    }
    injected: list[dict[str, Any]] = []
    reused: list[str] = []
    moved: list[dict[str, str]] = []
    injected_mount_frames: list[dict[str, Any]] = []
    configured: list[dict[str, Any]] = []
    for camera_entry in camera_config["cameras"]:
        camera_name = str(camera_entry["name"])
        role = str(camera_entry.get("role") or camera_name)
        mount = camera_entry.get("mount") or {"type": "world"}
        mount_type = str(mount.get("type") or "world")
        fovy = _camera_fovy(camera_entry)
        if mount_type == "world":
            extrinsics_type = str((camera_entry.get("extrinsics") or {}).get("type") or "")
            if extrinsics_type == "table_relative_lookat":
                position, target = _table_relative_camera_pose(camera_entry, placement)
                reference_metadata = {
                    "extrinsics_type": extrinsics_type,
                    "reference_table_center_xy": list(placement.table_center_xy),
                    "reference_table_top_z": float(placement.table_top_z),
                }
            elif extrinsics_type == "body_frame_relative_lookat":
                position, target, reference_metadata = _body_frame_relative_camera_pose(
                    camera_entry,
                    documents,
                    parents,
                )
            else:
                raise ValueError(
                    f"Unsupported world camera extrinsics type {extrinsics_type!r} "
                    f"for camera {camera_name!r}."
                )
            xyaxes = _camera_xyaxes(position, target)
            if camera_name in existing_by_name:
                reused.append(camera_name)
                camera_elem = existing_by_name[camera_name]
                mark_document_modified(element_documents.get(camera_elem))
                source = "reconfigured_existing_camera"
            else:
                camera_elem = ET.SubElement(worldbody, "camera", {"name": camera_name})
                existing_by_name[camera_name] = camera_elem
                element_documents[camera_elem] = documents[0]
                injected.append({"name": camera_name, "mode": "fixed", "mount": "world"})
                source = "injected_camera"
            camera_elem.set("mode", "fixed")
            camera_elem.set("fovy", f"{fovy:.9g}")
            camera_elem.set("pos", _format_float_sequence(position))
            camera_elem.set("xyaxes", _format_float_sequence(xyaxes))
            camera_elem.attrib.pop("quat", None)
            camera_elem.attrib.pop("euler", None)
            configured.append(
                {
                    "name": camera_name,
                    "role": role,
                    "mount": "world",
                    "mode": "fixed",
                    "pos": position.tolist(),
                    "target": target.tolist(),
                    "xyaxes": xyaxes.tolist(),
                    "fovy": fovy,
                    "intrinsics": deepcopy(camera_entry.get("intrinsics") or {}),
                    "extrinsics": deepcopy(camera_entry.get("extrinsics") or {}),
                    "resolved_reference_frame": reference_metadata,
                    "randomization": deepcopy(camera_entry.get("randomization") or {}),
                    "source": source,
                }
            )
            continue

        if mount_type == "body":
            mount_body = str(mount.get("body") or "").strip()
            if not mount_body:
                raise ValueError(f"Body-mounted camera {camera_name!r} must define mount.body.")
            target_body, target_document, mount_frame_metadata = _resolve_camera_mount_body(
                documents,
                mount,
                parents,
                element_documents,
            )
            resolved_mount_body = target_body.get("name") or mount_body
            if mount_frame_metadata["injected"]:
                injected_mount_frames.append(deepcopy(mount_frame_metadata))
            pos = _parse_vec(_format_float_sequence(mount.get("pos") or [0.0, 0.0, 0.0]), length=3)
            quat = _parse_vec(_format_float_sequence(mount.get("quat") or [1.0, 0.0, 0.0, 0.0]), length=4)
            if camera_name in existing_by_name:
                reused.append(camera_name)
                camera_elem = existing_by_name[camera_name]
                camera_document = element_documents.get(camera_elem)
                current_parent = parents.get(camera_elem)
                if current_parent is not None and current_parent is not target_body:
                    current_parent.remove(camera_elem)
                    target_body.append(camera_elem)
                    parents[camera_elem] = target_body
                    mark_document_modified(camera_document)
                    element_documents[camera_elem] = target_document
                    moved.append(
                        {
                            "name": camera_name,
                            "from": current_parent.get("name") or current_parent.tag,
                            "to": resolved_mount_body,
                        }
                    )
                mark_document_modified(element_documents.get(camera_elem))
                source = "reconfigured_existing_camera"
            else:
                camera_elem = ET.SubElement(target_body, "camera", {"name": camera_name})
                existing_by_name[camera_name] = camera_elem
                element_documents[camera_elem] = target_document
                injected.append({"name": camera_name, "mode": "fixed", "mount": f"body:{resolved_mount_body}"})
                source = "injected_camera"
            mark_document_modified(target_document)
            camera_elem.set("mode", "fixed")
            camera_elem.set("fovy", f"{fovy:.9g}")
            camera_elem.set("pos", _format_float_sequence(pos))
            camera_elem.set("quat", _format_float_sequence(quat))
            camera_elem.attrib.pop("xyaxes", None)
            camera_elem.attrib.pop("euler", None)
            configured.append(
                {
                    "name": camera_name,
                    "role": role,
                    "mount": "body",
                    "mount_body": mount_body,
                    "resolved_mount_body": resolved_mount_body,
                    "mode": "fixed",
                    "local_pos": pos.tolist(),
                    "local_quat": quat.tolist(),
                    "mount_frame_resolution": deepcopy(mount_frame_metadata),
                    "fovy": fovy,
                    "intrinsics": deepcopy(camera_entry.get("intrinsics") or {}),
                    "extrinsics": {"mount": deepcopy(mount)},
                    "randomization": deepcopy(camera_entry.get("randomization") or {}),
                    "source": source,
                }
            )
            continue

        raise ValueError(f"Unsupported camera mount type {mount_type!r} for camera {camera_name!r}.")
    modified_include_files = _write_modified_include_documents(documents, modified_include_paths)
    return {
        "camera_config": camera_config,
        "requested_cameras": _camera_names_from_config(camera_config),
        "reused_cameras": reused,
        "injected_cameras": injected,
        "moved_cameras": moved,
        "injected_camera_mount_frames": injected_mount_frames,
        "configured_cameras": configured,
        "camera_randomization": _camera_randomization_by_name(camera_config),
        "modified_mjcf_include_files": modified_include_files,
        "resolved_mjcf_include_file_attr_count": resolved_mjcf_include_file_attr_count,
    }


def _strip_invisible_collision_mesh_geoms(root: ET.Element) -> dict[str, Any]:
    """Remove transparent collision meshes that are irrelevant to state replay rendering.

    AXIS replay writes qpos / qvel / ctrl directly and only consumes RGB plus
    kinematic state. Transparent collision meshes therefore cannot affect the
    rendered observations, but MuJoCo still builds their convex hulls while
    compiling the model. A small number of backend meshes make that compile
    consume hundreds of GiB, so omit only geoms that are explicitly fully
    transparent and assigned to the collision group.
    """

    removed_names: list[str] = []
    for parent in root.iter():
        for geom in list(parent):
            if geom.tag != "geom" or geom.get("type") != "mesh":
                continue
            if geom.get("group", "0") != "0":
                continue
            rgba = geom.get("rgba", "").split()
            if len(rgba) != 4:
                continue
            try:
                alpha = float(rgba[3])
            except ValueError:
                continue
            if abs(alpha) > 1e-12:
                continue
            removed_names.append(geom.get("name") or geom.get("mesh") or "")
            parent.remove(geom)
    return {
        "state_replay_collision_mesh_policy": {
            "policy": "omit_fully_transparent_group_0_mesh_geoms",
            "removed_geom_count": len(removed_names),
            "removed_geom_names": removed_names,
        }
    }
