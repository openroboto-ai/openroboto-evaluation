from __future__ import annotations
import hashlib
import math
from typing import Any
import numpy as np
import mujoco

_DOMAIN_RANDOMIZATION_MISSING = object()


_MASK32 = (1 << 32) - 1


_U32_DENOM = float(1 << 32)


_DEFAULT_FREE_JOINT_POS_DELTA: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-0.05, 0.05),
    (-0.05, 0.05),
    (0.0, 0.0),
)


_DEFAULT_FREE_JOINT_EXCLUDED_PREFIXES = ("franka", "panda", "google_robot")


def _normalize_name_variants(raw: str | None) -> list[str]:
    if not raw:
        return []
    base = str(raw).strip()
    if not base:
        return []
    stripped = base.rstrip("/")
    variants = [base, stripped]
    if stripped:
        variants.append(f"{stripped}/")
        underscored = stripped.replace(" ", "_")
        spaced = stripped.replace("_", " ")
        variants.extend([underscored, f"{underscored}/", spaced, f"{spaced}/"])
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _resolve_body_id(model: mujoco.MjModel, entity_key: str, cfg: dict[str, Any]) -> int:
    mjcf = cfg.get("mjcf") or {}
    candidates: list[str] = []
    for raw in (
        mjcf.get("body"),
        mjcf.get("bodyName"),
        "franka/" if entity_key == "franka" else None,
        f"{entity_key}/",
        f"{entity_key}_base/",
        entity_key,
    ):
        candidates.extend(_normalize_name_variants(raw))
    for name in candidates:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return int(body_id)
    return -1


def _find_free_joint_for_body(model: mujoco.MjModel, body_id: int) -> int | None:
    for joint_id in range(model.njnt):
        if (
            int(model.jnt_bodyid[joint_id]) == body_id
            and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ):
            return joint_id
    return None


def _mocap_id_for_body(model: mujoco.MjModel, body_id: int) -> int | None:
    if body_id < 0 or body_id >= model.nbody:
        return None
    mocap_id = int(model.body_mocapid[body_id])
    return mocap_id if mocap_id >= 0 else None


def _parse_range(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            lo = float(value[0])
            hi = float(value[1])
        except (TypeError, ValueError):
            return None
        return lo, hi
    if isinstance(value, dict):
        try:
            lo = float(value["min"])
            hi = float(value["max"])
        except (KeyError, TypeError, ValueError):
            return None
        return lo, hi
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar, scalar


def _parse_vec3_ranges(value: Any) -> list[tuple[float, float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    parsed: list[tuple[float, float]] = []
    for item in value:
        rng = _parse_range(item)
        if rng is None:
            return None
        parsed.append(rng)
    return parsed


def _normalize_submit_nonce_text(submit_nonce: int | str) -> str:
    text = str(submit_nonce).strip()
    if not text or not text.isdigit():
        raise ValueError("submit_nonce must be an 8-digit numeric string")
    value = int(text)
    if value < 0 or value >= 100_000_000:
        raise ValueError("submit_nonce must be between 00000000 and 99999999")
    return f"{value:08d}"


def _mulberry32_next(state: int) -> tuple[int, float]:
    state = (state + 0x6D2B79F5) & _MASK32
    t = state
    t = ((t ^ (t >> 15)) * (t | 1)) & _MASK32
    t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & _MASK32)) & _MASK32
    out = (t ^ (t >> 14)) & _MASK32
    return state, out / _U32_DENOM


def _random_in_range(state: int, rng: tuple[float, float]) -> tuple[int, float]:
    state, unit = _mulberry32_next(state)
    lo, hi = rng
    return state, lo + unit * (hi - lo)


def _quaternion_from_euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx = math.cos(rx / 2.0)
    sx = math.sin(rx / 2.0)
    cy = math.cos(ry / 2.0)
    sy = math.sin(ry / 2.0)
    cz = math.cos(rz / 2.0)
    sz = math.sin(rz / 2.0)
    qw = cx * cy * cz - sx * sy * sz
    qx = sx * cy * cz + cx * sy * sz
    qy = cx * sy * cz - sx * cy * sz
    qz = cx * cy * sz + sx * sy * cz
    quat = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(quat)
    return quat / norm if norm > 0 else quat


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    out = np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(out)
    return out / norm if norm > 0 else out


def _should_apply_default_free_joint_randomization(body_name: str | None) -> bool:
    if not body_name:
        return False
    lowered = body_name.strip().lower()
    if not lowered:
        return False
    return not lowered.startswith(_DEFAULT_FREE_JOINT_EXCLUDED_PREFIXES)


def _apply_default_free_joint_randomization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state: int,
) -> int:
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not _should_apply_default_free_joint_randomization(body_name):
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        sampled = []
        for rng in _DEFAULT_FREE_JOINT_POS_DELTA:
            state, value = _random_in_range(state, rng)
            sampled.append(value)
        data.qpos[qadr:qadr + 3] = np.asarray(data.qpos[qadr:qadr + 3], dtype=np.float64) + np.asarray(
            sampled,
            dtype=np.float64,
        )
    return state


def _apply_domain_randomization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: dict[str, Any] | None | object,
    submit_nonce: int | str,
) -> None:
    state = int(_normalize_submit_nonce_text(submit_nonce)) & _MASK32
    if cfg is _DOMAIN_RANDOMIZATION_MISSING or cfg is None:
        _apply_default_free_joint_randomization(model, data, state)
        mujoco.mj_forward(model, data)
        return
    if not isinstance(cfg, dict) or not cfg:
        return

    def apply_entity(entity_key: str, entity_cfg: dict[str, Any]) -> int | None:
        nonlocal state
        body_id = _resolve_body_id(model, entity_key, entity_cfg)
        pos_ranges = _parse_vec3_ranges(entity_cfg.get("pos_delta"))
        rot_ranges = _parse_vec3_ranges(entity_cfg.get("rot_delta"))
        if body_id >= 0 and (pos_ranges or rot_ranges):
            joint_id = _find_free_joint_for_body(model, body_id)
            if joint_id is not None:
                qadr = int(model.jnt_qposadr[joint_id])
                if pos_ranges:
                    current = np.asarray(data.qpos[qadr:qadr + 3], dtype=np.float64)
                    sampled = []
                    for rng in pos_ranges:
                        state, value = _random_in_range(state, rng)
                        sampled.append(value)
                    sampled_pos = np.asarray(sampled, dtype=np.float64)
                    data.qpos[qadr:qadr + 3] = current + sampled_pos
                if rot_ranges:
                    current_quat = np.asarray(data.qpos[qadr + 3:qadr + 7], dtype=np.float64)
                    sampled_rot = []
                    for rng in rot_ranges:
                        state, value = _random_in_range(state, rng)
                        sampled_rot.append(value)
                    delta_quat = _quaternion_from_euler_xyz(*sampled_rot)
                    data.qpos[qadr + 3:qadr + 7] = _quat_multiply(current_quat, delta_quat)
            else:
                mocap_id = _mocap_id_for_body(model, body_id)
                if mocap_id is not None:
                    if pos_ranges:
                        current = np.asarray(data.mocap_pos[mocap_id], dtype=np.float64)
                        sampled = []
                        for rng in pos_ranges:
                            state, value = _random_in_range(state, rng)
                            sampled.append(value)
                        data.mocap_pos[mocap_id] = current + np.asarray(sampled, dtype=np.float64)
                    if rot_ranges:
                        current_quat = np.asarray(data.mocap_quat[mocap_id], dtype=np.float64)
                        sampled_rot = []
                        for rng in rot_ranges:
                            state, value = _random_in_range(state, rng)
                            sampled_rot.append(value)
                        delta_quat = _quaternion_from_euler_xyz(*sampled_rot)
                        data.mocap_quat[mocap_id] = _quat_multiply(current_quat, delta_quat)
        return body_id if body_id >= 0 else None

    objects = cfg.get("objects") if isinstance(cfg.get("objects"), dict) else {}

    for entity_key, entity_cfg in objects.items():
        if isinstance(entity_cfg, dict):
            apply_entity(str(entity_key), entity_cfg)

    swap_cfg = cfg.get("swap_positions")
    if swap_cfg:
        pairs = swap_cfg if isinstance(swap_cfg, list) else swap_cfg.get("pairs")
        try:
            probability = float((swap_cfg or {}).get("probability", 0.5)) if isinstance(swap_cfg, dict) else 0.5
        except (TypeError, ValueError):
            probability = 0.5
        if isinstance(pairs, list):
            for pair in pairs:
                if not isinstance(pair, list) or len(pair) < 2:
                    continue
                state, sampled = _random_in_range(state, (0.0, 1.0))
                if sampled > probability:
                    continue
                a_cfg = objects.get(pair[0])
                b_cfg = objects.get(pair[1])
                if not isinstance(a_cfg, dict) or not isinstance(b_cfg, dict):
                    continue
                a_body = _resolve_body_id(model, str(pair[0]), a_cfg)
                b_body = _resolve_body_id(model, str(pair[1]), b_cfg)
                if a_body < 0 or b_body < 0:
                    continue
                a_joint = _find_free_joint_for_body(model, a_body)
                b_joint = _find_free_joint_for_body(model, b_body)
                if a_joint is None or b_joint is None:
                    continue
                a_qadr = int(model.jnt_qposadr[a_joint])
                b_qadr = int(model.jnt_qposadr[b_joint])
                a_dadr = int(model.jnt_dofadr[a_joint])
                b_dadr = int(model.jnt_dofadr[b_joint])
                a_qpos = np.copy(data.qpos[a_qadr:a_qadr + 7])
                data.qpos[a_qadr:a_qadr + 7] = data.qpos[b_qadr:b_qadr + 7]
                data.qpos[b_qadr:b_qadr + 7] = a_qpos
                a_qvel = np.copy(data.qvel[a_dadr:a_dadr + 6])
                data.qvel[a_dadr:a_dadr + 6] = data.qvel[b_dadr:b_dadr + 6]
                data.qvel[b_dadr:b_dadr + 6] = a_qvel

    mujoco.mj_forward(model, data)
