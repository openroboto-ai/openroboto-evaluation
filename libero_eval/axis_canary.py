"""Small, auditable AXIS trajectory-canary checkpoint runtime.

The canary deliberately memorizes one qualified expert trajectory.  It is a
deployment health check for the simulator, policy transport, action adapter,
and success checker; it is not a learned VLA and must never be used for model
ranking.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
ACTION_DIM = 9


@dataclasses.dataclass(frozen=True)
class AxisCanaryCheckpoint:
    metadata: dict[str, Any]
    initial_state: np.ndarray
    actions: np.ndarray


def checkpoint_digest(initial_state: np.ndarray, actions: np.ndarray, metadata: dict[str, Any]) -> str:
    """Return a stable digest over the checkpoint's semantic contents."""
    hasher = hashlib.sha256()
    hasher.update(np.asarray(initial_state, dtype="<f4").tobytes(order="C"))
    hasher.update(np.asarray(actions, dtype="<f4").tobytes(order="C"))
    public_metadata = {key: value for key, value in metadata.items() if key != "checkpoint_sha256"}
    hasher.update(
        json.dumps(public_metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return hasher.hexdigest()


def save_checkpoint(
    path: pathlib.Path,
    *,
    metadata: dict[str, Any],
    initial_state: np.ndarray,
    actions: np.ndarray,
) -> AxisCanaryCheckpoint:
    initial_state = np.asarray(initial_state, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    validated = _validate(metadata, initial_state, actions, require_digest=False)
    metadata = dict(validated.metadata)
    metadata["checkpoint_sha256"] = checkpoint_digest(initial_state, actions, metadata)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        initial_state=initial_state,
        actions=actions,
    )
    return _validate(metadata, initial_state, actions, require_digest=True)


def load_checkpoint(path: pathlib.Path) -> AxisCanaryCheckpoint:
    path = pathlib.Path(path)
    try:
        with np.load(path, allow_pickle=False) as payload:
            metadata_raw = payload["metadata_json"].item()
            initial_state = np.asarray(payload["initial_state"], dtype=np.float32)
            actions = np.asarray(payload["actions"], dtype=np.float32)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid AXIS canary checkpoint {path}: {exc}") from exc
    try:
        metadata = json.loads(str(metadata_raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid AXIS canary metadata in {path}: {exc}") from exc
    return _validate(metadata, initial_state, actions, require_digest=True)


def _validate(
    metadata: dict[str, Any],
    initial_state: np.ndarray,
    actions: np.ndarray,
    *,
    require_digest: bool,
) -> AxisCanaryCheckpoint:
    if not isinstance(metadata, dict):
        raise ValueError("AXIS canary metadata must be an object")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported AXIS canary schema_version {metadata.get('schema_version')!r}")
    if metadata.get("purpose") != "deployment-canary-only":
        raise ValueError("AXIS canary purpose must be 'deployment-canary-only'")
    if not isinstance(metadata.get("task_id"), int):
        raise ValueError("AXIS canary task_id must be an integer")
    if not isinstance(metadata.get("instruction"), str) or not metadata["instruction"].strip():
        raise ValueError("AXIS canary instruction must be a non-empty string")
    if initial_state.shape != (ACTION_DIM,) or not np.isfinite(initial_state).all():
        raise ValueError(f"AXIS canary initial_state must contain {ACTION_DIM} finite values")
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"AXIS canary actions must have shape [T,{ACTION_DIM}], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("AXIS canary actions contain non-finite values")
    expected = metadata.get("checkpoint_sha256")
    if require_digest:
        actual = checkpoint_digest(initial_state, actions, metadata)
        if not isinstance(expected, str) or expected != actual:
            raise ValueError(f"AXIS canary digest mismatch: got {actual}, expected {expected!r}")
    return AxisCanaryCheckpoint(dict(metadata), initial_state.copy(), actions.copy())


class AxisCanaryPolicy:
    """Stateful fixed-trajectory policy used by the WebSocket canary server."""

    def __init__(self, checkpoint: AxisCanaryCheckpoint, chunk_size: int = 5) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.checkpoint = checkpoint
        self.chunk_size = int(chunk_size)
        self._cursor = 0
        self._requests = 0

    def reset(self) -> None:
        self._cursor = 0
        self._requests = 0

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        prompt = observation.get("prompt")
        if prompt != self.checkpoint.metadata["instruction"]:
            raise ValueError(
                f"canary checkpoint is only valid for {self.checkpoint.metadata['instruction']!r}, got {prompt!r}"
            )
        state = np.asarray(observation.get("observation/state"), dtype=np.float32)
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"observation/state must contain {ACTION_DIM} finite values")

        # A fresh evaluator trial returns to the frozen robot reset pose.  The
        # check is intentionally strict enough not to reset during ordinary
        # trajectory motion, while tolerating MuJoCo's post-settle drift.
        reset_tolerance = float(self.checkpoint.metadata.get("reset_l2_tolerance", 0.02))
        if self._requests and np.linalg.norm(state - self.checkpoint.initial_state) <= reset_tolerance:
            self.reset()

        start = self._cursor
        stop = min(len(self.checkpoint.actions), start + self.chunk_size)
        if start >= len(self.checkpoint.actions):
            chunk = self.checkpoint.actions[-1:]
        else:
            chunk = self.checkpoint.actions[start:stop]
            self._cursor = stop
        self._requests += 1
        return {
            "actions": chunk.copy(),
            "canary": {
                "cursor_start": start,
                "cursor_stop": self._cursor,
                "deployment_canary_only": True,
            },
        }
