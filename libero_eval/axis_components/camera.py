from __future__ import annotations
import hashlib
import math
from typing import Any
import numpy as np
import mujoco

DEFAULT_TABLE_FULL_SIZE = (1.4, 1.2, 0.05)


def stable_seed(global_seed: int, *parts: Any) -> int:
    digest = hashlib.sha256()
    digest.update(str(int(global_seed)).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False) % (2**32)


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


def _quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    normalized = _normalize_quat_wxyz(quat, label="matrix conversion quaternion")
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, normalized)
    return matrix.reshape(3, 3)


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    resolved = np.asarray(matrix, dtype=np.float64)
    if resolved.shape != (3, 3):
        raise ValueError(f"Rotation matrix must be 3x3, got {resolved.shape}.")
    if not np.allclose(resolved.T @ resolved, np.eye(3), atol=1e-8) or not np.isclose(
        np.linalg.det(resolved), 1.0, atol=1e-8
    ):
        raise ValueError("Rotation matrix must be orthonormal with determinant +1.")
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, resolved.reshape(-1))
    quat = _normalize_quat_wxyz(quat, label="rotation matrix quaternion")
    return -quat if quat[0] < 0.0 else quat


def _rotation_x(angle_rad: float) -> np.ndarray:
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_y(angle_rad: float) -> np.ndarray:
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _rotation_z(angle_rad: float) -> np.ndarray:
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _camera_field_uniform(
    camera_seed: int,
    field_name: str,
    minimum: float,
    maximum: float,
) -> tuple[float, int]:
    field_seed = stable_seed(camera_seed, field_name)
    random_state = np.random.RandomState(field_seed)
    return float(random_state.uniform(float(minimum), float(maximum))), int(field_seed)


def _sample_wrist_optical_pose_box(
    *,
    base_position: np.ndarray,
    base_quat_wxyz: np.ndarray,
    base_fovy_deg: float,
    profile: dict[str, Any],
    camera_seed: int,
) -> dict[str, Any]:
    if str(profile.get("mode") or "") != WRIST_CAMERA_RANDOMIZATION_MODE:
        raise ValueError(f"Expected wrist randomization mode {WRIST_CAMERA_RANDOMIZATION_MODE!r}.")
    base_position = np.asarray(base_position, dtype=np.float64)
    if base_position.shape != (3,) or not np.all(np.isfinite(base_position)):
        raise ValueError("Wrist base camera position must have three finite values.")
    base_quat = _normalize_quat_wxyz(base_quat_wxyz, label="wrist base camera quaternion")
    base_rotation = _quat_to_matrix_wxyz(base_quat)
    position_ranges = profile["position_half_ranges_m"]
    orientation_ranges = profile["orientation_half_ranges_deg"]
    fovy_profile = profile["fovy_deg"]
    if not math.isclose(float(base_fovy_deg), float(fovy_profile["center"]), abs_tol=1e-9):
        raise ValueError(
            f"Wrist base fovy={base_fovy_deg} does not match configured center={fovy_profile['center']}."
        )

    sampled: dict[str, float] = {}
    field_seeds: dict[str, int] = {}
    for axis in ("depth", "lateral", "vertical"):
        half_range = float(position_ranges[axis])
        sampled[f"{axis}_m"], field_seeds[f"position_{axis}"] = _camera_field_uniform(
            camera_seed,
            f"position_{axis}",
            -half_range,
            half_range,
        )
    for axis in ("roll", "pitch", "yaw"):
        half_range = float(orientation_ranges[axis])
        sampled[f"{axis}_deg"], field_seeds[f"orientation_{axis}"] = _camera_field_uniform(
            camera_seed,
            f"orientation_{axis}",
            -half_range,
            half_range,
        )
    fovy_delta, field_seeds["fovy"] = _camera_field_uniform(
        camera_seed,
        "fovy",
        -float(fovy_profile["half_range"]),
        float(fovy_profile["half_range"]),
    )
    resolved_fovy = float(fovy_profile["center"]) + fovy_delta

    # axis-training uses the right-handed convention (+X forward, +Z up), so
    # its +Y lateral axis points camera-left. MuJoCo/OpenGL camera coordinates
    # are (+X image-right, +Y image-up, -Z forward).
    dr_basis_to_mujoco = np.asarray(
        [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    if not math.isclose(float(np.linalg.det(dr_basis_to_mujoco)), 1.0, abs_tol=1e-12):
        raise ValueError("Camera DR basis conversion must be a proper right-handed rotation.")
    position_delta_dr = np.asarray(
        [sampled["depth_m"], sampled["lateral_m"], sampled["vertical_m"]],
        dtype=np.float64,
    )
    position_delta_camera = dr_basis_to_mujoco @ position_delta_dr
    position_delta_parent = base_rotation @ position_delta_camera
    resolved_position = base_position + position_delta_parent

    roll = math.radians(sampled["roll_deg"])
    pitch = math.radians(sampled["pitch_deg"])
    yaw = math.radians(sampled["yaw_deg"])
    local_rotation_dr = _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)
    local_rotation_mujoco = dr_basis_to_mujoco @ local_rotation_dr @ dr_basis_to_mujoco.T
    resolved_rotation = base_rotation @ local_rotation_mujoco
    resolved_quat = _matrix_to_quat_wxyz(resolved_rotation)

    return {
        "mode": WRIST_CAMERA_RANDOMIZATION_MODE,
        "camera_seed": int(camera_seed),
        "field_seeds": field_seeds,
        "sampling_frequency": "once_per_trajectory_variant",
        "coordinate_frame": "camera_optical_local_then_panda_link8_local",
        "base": {
            "position": base_position.tolist(),
            "quat_wxyz": base_quat.tolist(),
            "fovy_deg": float(base_fovy_deg),
        },
        "sampled": sampled | {"fovy_delta_deg": float(fovy_delta)},
        "resolved": {
            "position": resolved_position.tolist(),
            "quat_wxyz": resolved_quat.tolist(),
            "fovy_deg": resolved_fovy,
            "position_delta_camera_mujoco_xyz": position_delta_camera.tolist(),
            "position_delta_parent": position_delta_parent.tolist(),
            "rotation_delta_camera_dr_basis": local_rotation_dr.tolist(),
            "rotation_delta_camera_mujoco": local_rotation_mujoco.tolist(),
            "rotation_delta_camera_mujoco_quat_wxyz": _matrix_to_quat_wxyz(
                local_rotation_mujoco
            ).tolist(),
        },
    }


def _front_table_sector_geometry(
    *,
    base_world_position: np.ndarray,
    base_world_quat_wxyz: np.ndarray,
    table_center_xy_world: tuple[float, float] | list[float],
    table_top_z_world: float,
    table_full_size: tuple[float, float, float] | list[float],
) -> dict[str, Any]:
    base_position = np.asarray(base_world_position, dtype=np.float64)
    base_quat = _normalize_quat_wxyz(base_world_quat_wxyz, label="front reference base quaternion")
    table_center = np.asarray(table_center_xy_world, dtype=np.float64)
    full_size = np.asarray(table_full_size, dtype=np.float64)
    if base_position.shape != (3,) or table_center.shape != (2,) or full_size.shape != (3,):
        raise ValueError("Front camera geometry requires base xyz, table center xy, and table full size xyz.")
    if not np.all(np.isfinite(base_position)) or not np.all(np.isfinite(table_center)):
        raise ValueError("Front camera geometry inputs must be finite.")
    if np.any(full_size <= 0.0) or not np.all(np.isfinite(full_size)):
        raise ValueError("Front camera table full size must contain three positive finite values.")
    base_up_world = _quat_rotate_wxyz(base_quat, np.asarray([0.0, 0.0, 1.0]))
    if not np.allclose(base_up_world, [0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(
            "Front table-sector randomization requires an upright Franka base; yaw is supported, "
            f"but the resolved base +Z axis is {base_up_world.tolist()}."
        )
    inverse_base_quat = base_quat * np.asarray([1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    half_x = float(full_size[0]) * 0.5
    half_y = float(full_size[1]) * 0.5
    world_corners = np.asarray(
        [
            [table_center[0] - half_x, table_center[1] - half_y, table_top_z_world],
            [table_center[0] - half_x, table_center[1] + half_y, table_top_z_world],
            [table_center[0] + half_x, table_center[1] + half_y, table_top_z_world],
            [table_center[0] + half_x, table_center[1] - half_y, table_top_z_world],
        ],
        dtype=np.float64,
    )
    base_corners = np.stack(
        [_quat_rotate_wxyz(inverse_base_quat, corner - base_position) for corner in world_corners],
        axis=0,
    )
    edge_indexes = ((0, 1), (1, 2), (2, 3), (3, 0))
    selected_edge = max(
        edge_indexes,
        key=lambda edge: float(np.mean(base_corners[list(edge), 0])),
    )
    edge_start = base_corners[selected_edge[0], :2]
    edge_end = base_corners[selected_edge[1], :2]
    if float(np.mean([edge_start[0], edge_end[0]])) <= 0.0:
        raise ValueError("The table has no positive-X edge in the configured Franka base frame.")
    start_angle = math.atan2(float(edge_start[1]), float(edge_start[0]))
    end_angle = math.atan2(float(edge_end[1]), float(edge_end[0]))
    short_arc = math.atan2(math.sin(end_angle - start_angle), math.cos(end_angle - start_angle))
    if abs(short_arc) <= 1e-8 or abs(short_arc) >= math.pi:
        raise ValueError("Table far-edge corner rays do not define a valid short azimuth sector.")
    return {
        "base_world_position": base_position.tolist(),
        "base_world_quat_wxyz": base_quat.tolist(),
        "table_world_corners": world_corners.tolist(),
        "table_corners_in_reference_frame": base_corners.tolist(),
        "far_edge_corner_indexes": list(selected_edge),
        "far_edge_start_xy": edge_start.tolist(),
        "far_edge_end_xy": edge_end.tolist(),
        "azimuth_start_rad": start_angle,
        "azimuth_short_arc_rad": short_arc,
        "azimuth_end_unwrapped_rad": start_angle + short_arc,
        "azimuth_corner_degrees": [math.degrees(start_angle), math.degrees(start_angle + short_arc)],
    }


def _ray_segment_intersection_distance(
    direction_xy: np.ndarray,
    segment_start_xy: np.ndarray,
    segment_end_xy: np.ndarray,
) -> tuple[float, float]:
    direction = np.asarray(direction_xy, dtype=np.float64)
    start = np.asarray(segment_start_xy, dtype=np.float64)
    end = np.asarray(segment_end_xy, dtype=np.float64)
    matrix = np.column_stack([direction, -(end - start)])
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) <= 1e-10:
        raise ValueError("Camera azimuth ray is parallel to the selected table edge.")
    distance, segment_fraction = np.linalg.solve(matrix, start)
    if distance <= 0.0 or segment_fraction < -1e-7 or segment_fraction > 1.0 + 1e-7:
        raise ValueError(
            "Camera azimuth ray did not intersect the selected table edge segment: "
            f"distance={distance}, segment_fraction={segment_fraction}."
        )
    return float(distance), float(np.clip(segment_fraction, 0.0, 1.0))


def _front_workspace_points_in_reference_frame(profile: dict[str, Any]) -> np.ndarray:
    visibility = profile["visibility_gate"]
    center = np.asarray(visibility["workspace_center_offset_in_reference_frame_m"], dtype=np.float64)
    radius = float(visibility["horizontal_radius_m"])
    height_min, height_max = (float(value) for value in visibility["height_range_in_reference_frame_m"])
    count = int(visibility["azimuth_samples"])
    points: list[np.ndarray] = []
    for height in (height_min, height_max):
        resolved_height = float(center[2]) + height
        points.append(np.asarray([center[0], center[1], resolved_height], dtype=np.float64))
        for angle in np.linspace(0.0, 2.0 * math.pi, count, endpoint=False):
            points.append(
                np.asarray(
                    [
                        center[0] + radius * math.cos(angle),
                        center[1] + radius * math.sin(angle),
                        resolved_height,
                    ],
                    dtype=np.float64,
                )
            )
    return np.stack(points, axis=0)


def _camera_visible_point_count(
    *,
    camera_position: np.ndarray,
    look_at: np.ndarray,
    fovy_deg: float,
    aspect_ratio: float,
    points: np.ndarray,
) -> int:
    position = np.asarray(camera_position, dtype=np.float64)
    target = np.asarray(look_at, dtype=np.float64)
    forward = target - position
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    xyaxes = _camera_xyaxes(position, target)
    right = xyaxes[:3]
    up = xyaxes[3:]
    vertical_limit = math.radians(float(fovy_deg)) * 0.5
    horizontal_limit = math.atan(math.tan(vertical_limit) * float(aspect_ratio))
    visible = 0
    for point in np.asarray(points, dtype=np.float64):
        relative = point - position
        depth = float(np.dot(relative, forward))
        if depth <= 1e-8:
            continue
        horizontal_angle = abs(math.atan2(float(np.dot(relative, right)), depth))
        vertical_angle = abs(math.atan2(float(np.dot(relative, up)), depth))
        if horizontal_angle <= horizontal_limit and vertical_angle <= vertical_limit:
            visible += 1
    return visible


def _lookat_quat_wxyz(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    xyaxes = _camera_xyaxes(position, target)
    right = xyaxes[:3]
    up = xyaxes[3:]
    backward = np.cross(right, up)
    return _matrix_to_quat_wxyz(np.column_stack([right, up, backward]))


def _sample_front_base_table_sector_lookat(
    *,
    profile: dict[str, Any],
    camera_seed: int,
    base_world_position: np.ndarray,
    base_world_quat_wxyz: np.ndarray,
    table_center_xy_world: tuple[float, float] | list[float],
    table_top_z_world: float,
    table_full_size: tuple[float, float, float] | list[float],
) -> dict[str, Any]:
    if str(profile.get("mode") or "") != FRONT_CAMERA_RANDOMIZATION_MODE:
        raise ValueError(f"Expected front randomization mode {FRONT_CAMERA_RANDOMIZATION_MODE!r}.")
    geometry = _front_table_sector_geometry(
        base_world_position=base_world_position,
        base_world_quat_wxyz=base_world_quat_wxyz,
        table_center_xy_world=table_center_xy_world,
        table_top_z_world=table_top_z_world,
        table_full_size=table_full_size,
    )
    base_position = np.asarray(geometry["base_world_position"], dtype=np.float64)
    base_quat = np.asarray(geometry["base_world_quat_wxyz"], dtype=np.float64)
    edge_start = np.asarray(geometry["far_edge_start_xy"], dtype=np.float64)
    edge_end = np.asarray(geometry["far_edge_end_xy"], dtype=np.float64)
    start_angle = float(geometry["azimuth_start_rad"])
    short_arc = float(geometry["azimuth_short_arc_rad"])
    clearance_profile = profile["table_edge_clearance_m"]
    height_profile = profile["height_offset_in_reference_frame_m"]
    look_at_profile = profile["look_at"]
    fovy_profile = profile["fovy_deg"]
    visibility_profile = profile["visibility_gate"]
    aspect_ratio = float(visibility_profile["render_aspect_ratio"][0]) / float(
        visibility_profile["render_aspect_ratio"][1]
    )
    workspace_points_reference = _front_workspace_points_in_reference_frame(profile)
    workspace_points_world = np.stack(
        [base_position + _quat_rotate_wxyz(base_quat, point) for point in workspace_points_reference],
        axis=0,
    )
    best_visible_count = -1
    total_points = int(workspace_points_world.shape[0])
    for attempt_index in range(int(visibility_profile["max_attempts"])):
        prefix = f"attempt_{attempt_index}"
        field_seeds: dict[str, int] = {}
        azimuth_fraction, field_seeds["azimuth"] = _camera_field_uniform(
            camera_seed, f"{prefix}_azimuth", 0.0, 1.0
        )
        azimuth_unwrapped = start_angle + short_arc * azimuth_fraction
        azimuth = math.atan2(math.sin(azimuth_unwrapped), math.cos(azimuth_unwrapped))
        direction = np.asarray([math.cos(azimuth), math.sin(azimuth)], dtype=np.float64)
        edge_distance, edge_segment_fraction = _ray_segment_intersection_distance(
            direction,
            edge_start,
            edge_end,
        )
        clearance, field_seeds["table_edge_clearance"] = _camera_field_uniform(
            camera_seed,
            f"{prefix}_table_edge_clearance",
            float(clearance_profile["minimum"]),
            float(clearance_profile["maximum"]),
        )
        height_delta, field_seeds["height"] = _camera_field_uniform(
            camera_seed,
            f"{prefix}_height",
            -float(height_profile["half_range"]),
            float(height_profile["half_range"]),
        )
        radius_unit, field_seeds["look_at_radius"] = _camera_field_uniform(
            camera_seed, f"{prefix}_look_at_radius", 0.0, 1.0
        )
        disk_angle, field_seeds["look_at_angle"] = _camera_field_uniform(
            camera_seed, f"{prefix}_look_at_angle", -math.pi, math.pi
        )
        fovy_delta, field_seeds["fovy"] = _camera_field_uniform(
            camera_seed,
            f"{prefix}_fovy",
            -float(fovy_profile["half_range"]),
            float(fovy_profile["half_range"]),
        )
        camera_radius = edge_distance + clearance
        camera_position_reference = np.asarray(
            [
                camera_radius * direction[0],
                camera_radius * direction[1],
                float(height_profile["center"]) + height_delta,
            ],
            dtype=np.float64,
        )
        disk_radius = float(look_at_profile["disk_diameter_m"]) * 0.5
        sampled_look_at_radius = disk_radius * math.sqrt(radius_unit)
        look_at_reference = np.asarray(
            look_at_profile["center_offset_in_reference_frame_m"], dtype=np.float64
        ).copy()
        look_at_reference[:2] += sampled_look_at_radius * np.asarray(
            [math.cos(disk_angle), math.sin(disk_angle)], dtype=np.float64
        )
        camera_position_world = base_position + _quat_rotate_wxyz(base_quat, camera_position_reference)
        look_at_world = base_position + _quat_rotate_wxyz(base_quat, look_at_reference)
        fovy = float(fovy_profile["center"]) + fovy_delta
        visible_count = _camera_visible_point_count(
            camera_position=camera_position_world,
            look_at=look_at_world,
            fovy_deg=fovy,
            aspect_ratio=aspect_ratio,
            points=workspace_points_world,
        )
        best_visible_count = max(best_visible_count, visible_count)
        if visible_count != total_points:
            continue
        camera_quat = _lookat_quat_wxyz(camera_position_world, look_at_world)
        camera_to_look_at_reference = look_at_reference - camera_position_reference
        return {
            "mode": FRONT_CAMERA_RANDOMIZATION_MODE,
            "camera_seed": int(camera_seed),
            "field_seeds": field_seeds,
            "sampling_frequency": "once_per_trajectory_variant",
            "candidate_distribution": "independent_configured_uniforms",
            "accepted_distribution": "candidate_distribution_conditioned_on_visibility_gate",
            "coordinate_frame": "franka_reference_base_then_world",
            "attempt_index": int(attempt_index),
            "visibility": {
                "required_point_count": total_points,
                "visible_point_count": visible_count,
                "workspace_points_in_reference_frame": workspace_points_reference.tolist(),
                "aspect_ratio": aspect_ratio,
            },
            "geometry": geometry,
            "sampled": {
                "azimuth_fraction": azimuth_fraction,
                "azimuth_rad": azimuth,
                "azimuth_deg": math.degrees(azimuth),
                "table_edge_intersection_distance_m": edge_distance,
                "table_edge_segment_fraction": edge_segment_fraction,
                "table_edge_clearance_m": clearance,
                "camera_horizontal_radius_from_base_m": camera_radius,
                "height_delta_m": height_delta,
                "look_at_disk_radius_m": sampled_look_at_radius,
                "look_at_disk_angle_rad": disk_angle,
                "fovy_delta_deg": fovy_delta,
                "camera_to_look_at_horizontal_distance_m": float(
                    np.linalg.norm(camera_to_look_at_reference[:2])
                ),
                "camera_to_look_at_distance_m": float(np.linalg.norm(camera_to_look_at_reference)),
            },
            "resolved": {
                "position_in_reference_frame": camera_position_reference.tolist(),
                "look_at_in_reference_frame": look_at_reference.tolist(),
                "world_position": camera_position_world.tolist(),
                "world_look_at": look_at_world.tolist(),
                "world_quat_wxyz": camera_quat.tolist(),
                "fovy_deg": fovy,
            },
        }
    raise ValueError(
        "Front camera visibility gate exhausted all deterministic candidates: "
        f"attempts={visibility_profile['max_attempts']}, "
        f"best_visible={best_visible_count}/{total_points}."
    )
