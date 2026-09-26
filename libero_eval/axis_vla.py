"""Runtime-aligned training artifact and OpenPI transforms for native AXIS.

The benchmark receives native 9D Franka joint observations and emits absolute
9D joint-position targets.  This module deliberately keeps that contract
unchanged between exported training samples and policy inference.
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
TRAINING_PURPOSE = "benchmark-validation-vla-training"


@dataclasses.dataclass(frozen=True)
class AxisVlaArtifact:
    metadata: dict[str, Any]
    images: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    episode_ends: np.ndarray


def artifact_digest(
    images: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    """Return a stable SHA-256 over the semantic artifact contents."""

    hasher = hashlib.sha256()
    for array, dtype in (
        (images, np.dtype("u1")),
        (states, np.dtype("<f4")),
        (actions, np.dtype("<f4")),
        (episode_ends, np.dtype("<i8")),
    ):
        normalized = np.ascontiguousarray(array, dtype=dtype)
        hasher.update(str(normalized.shape).encode("ascii"))
        hasher.update(normalized.tobytes(order="C"))
    public_metadata = {key: value for key, value in metadata.items() if key != "artifact_sha256"}
    hasher.update(
        json.dumps(public_metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return hasher.hexdigest()


def _validate(
    metadata: dict[str, Any],
    images: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
    *,
    require_digest: bool,
) -> AxisVlaArtifact:
    if not isinstance(metadata, dict):
        raise ValueError("AXIS VLA artifact metadata must be an object")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported AXIS VLA artifact schema_version {metadata.get('schema_version')!r}")
    if metadata.get("purpose") != TRAINING_PURPOSE:
        raise ValueError(f"AXIS VLA artifact purpose must be {TRAINING_PURPOSE!r}")
    if metadata.get("eligible_for_scoring") is not False:
        raise ValueError("runtime-replay benchmark-validation data must set eligible_for_scoring=false")
    if not isinstance(metadata.get("instruction"), str) or not metadata["instruction"].strip():
        raise ValueError("AXIS VLA artifact instruction must be a non-empty string")
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError(f"AXIS VLA images must be uint8 [T,H,W,3], got {images.shape} {images.dtype}")
    if states.ndim != 2 or states.shape[1] != ACTION_DIM or not np.isfinite(states).all():
        raise ValueError(f"AXIS VLA states must be finite [T,{ACTION_DIM}], got {states.shape}")
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not np.isfinite(actions).all():
        raise ValueError(f"AXIS VLA actions must be finite [T,{ACTION_DIM}], got {actions.shape}")
    if not (len(images) == len(states) == len(actions)) or not len(images):
        raise ValueError(
            f"AXIS VLA arrays must have the same non-zero length, got {len(images)}/{len(states)}/{len(actions)}"
        )
    if episode_ends.ndim != 1 or not len(episode_ends):
        raise ValueError("AXIS VLA episode_ends must be a non-empty vector")
    if np.any(episode_ends <= 0) or np.any(np.diff(episode_ends) <= 0) or int(episode_ends[-1]) != len(images):
        raise ValueError(f"invalid AXIS VLA episode_ends {episode_ends.tolist()} for {len(images)} frames")
    if int(metadata.get("training_examples", -1)) != len(images):
        raise ValueError(
            f"metadata training_examples={metadata.get('training_examples')!r} does not match {len(images)} frames"
        )
    if int(metadata.get("successful_replays", -1)) != len(episode_ends):
        raise ValueError(
            f"metadata successful_replays={metadata.get('successful_replays')!r} does not match "
            f"{len(episode_ends)} episodes"
        )
    instructions = metadata.get("episode_instructions")
    if instructions is not None and (
        not isinstance(instructions, list)
        or len(instructions) != len(episode_ends)
        or any(not isinstance(value, str) or not value.strip() for value in instructions)
    ):
        raise ValueError("episode_instructions must contain one non-empty instruction per episode")
    if require_digest:
        expected = metadata.get("artifact_sha256")
        actual = artifact_digest(images, states, actions, episode_ends, metadata)
        if not isinstance(expected, str) or expected != actual:
            raise ValueError(f"AXIS VLA artifact digest mismatch: got {actual}, expected {expected!r}")
    return AxisVlaArtifact(
        dict(metadata),
        np.asarray(images, dtype=np.uint8),
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(episode_ends, dtype=np.int64),
    )


def save_artifact(
    path: pathlib.Path,
    *,
    metadata: dict[str, Any],
    images: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
) -> AxisVlaArtifact:
    artifact = _validate(metadata, images, states, actions, episode_ends, require_digest=False)
    metadata = dict(artifact.metadata)
    metadata["artifact_sha256"] = artifact_digest(
        artifact.images, artifact.states, artifact.actions, artifact.episode_ends, metadata
    )
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        images=artifact.images,
        states=artifact.states,
        actions=artifact.actions,
        episode_ends=artifact.episode_ends,
    )
    return _validate(
        metadata, artifact.images, artifact.states, artifact.actions, artifact.episode_ends, require_digest=True
    )


def load_artifact(path: pathlib.Path) -> AxisVlaArtifact:
    path = pathlib.Path(path)
    try:
        with np.load(path, allow_pickle=False) as payload:
            metadata_raw = payload["metadata_json"].item()
            images = np.asarray(payload["images"])
            states = np.asarray(payload["states"], dtype=np.float32)
            actions = np.asarray(payload["actions"], dtype=np.float32)
            episode_ends = np.asarray(payload["episode_ends"], dtype=np.int64)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid AXIS VLA artifact {path}: {exc}") from exc
    try:
        metadata = json.loads(str(metadata_raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid AXIS VLA metadata in {path}: {exc}") from exc
    return _validate(metadata, images, states, actions, episode_ends, require_digest=True)


class AxisReplayDataset:
    """Frame-indexed dataset whose action chunks never cross episode boundaries."""

    def __init__(self, path: pathlib.Path, action_horizon: int) -> None:
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        self.artifact = load_artifact(path)
        self.action_horizon = int(action_horizon)

    def __len__(self) -> int:
        return len(self.artifact.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.artifact.episode_ends, index, side="right"))
        episode_end = int(self.artifact.episode_ends[episode])
        action_indices = np.minimum(index + np.arange(self.action_horizon), episode_end - 1)
        return {
            "observation/image": self.artifact.images[index],
            "observation/state": self.artifact.states[index],
            "actions": self.artifact.actions[action_indices],
            "prompt": (
                self.artifact.metadata["episode_instructions"][episode]
                if "episode_instructions" in self.artifact.metadata
                else self.artifact.metadata["instruction"]
            ),
        }


def _parse_image(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"AXIS observation/image must be rank 3, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"AXIS observation/image must have three channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(image, dtype=np.uint8)


@dataclasses.dataclass(frozen=True)
class AxisInputs:
    """Map evaluator-owned AXIS fields to Pi0/Pi0.5 model inputs."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        image = _parse_image(data["observation/image"])
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"AXIS observation/state must contain {ACTION_DIM} finite values, got {state.shape}")
        result: dict[str, Any] = {
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": np.zeros_like(image),
                "right_wrist_0_rgb": np.zeros_like(image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
            "state": state,
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not np.isfinite(actions).all():
                raise ValueError(f"AXIS training actions must be finite [T,{ACTION_DIM}], got {actions.shape}")
            result["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            result["prompt"] = prompt
        return result


@dataclasses.dataclass(frozen=True)
class AxisOutputs:
    """Decode unnormalized joint targets, with an optional symmetric gripper prior."""

    gripper_mode: str = "continuous"

    def __post_init__(self) -> None:
        if self.gripper_mode not in ("continuous", "symmetric-binary"):
            raise ValueError(f"unsupported AXIS gripper mode: {self.gripper_mode}")

    def __call__(self, data: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[1] < ACTION_DIM:
            raise ValueError(f"OpenPI actions must be [T,D] with D >= {ACTION_DIM}, got {actions.shape}")
        native = np.asarray(actions[:, :ACTION_DIM])
        if not np.isfinite(native).all():
            raise ValueError("OpenPI returned non-finite AXIS action targets")
        if self.gripper_mode == "symmetric-binary":
            # The replay demonstrations command both fingers together at 0 or
            # 0.04 m. Apply that prior uniformly, after OpenPI unnormalization;
            # it uses neither task identity nor checker feedback.
            native = native.astype(np.float32, copy=True)
            opened = native[:, :2].mean(axis=1) >= 0.02
            native[:, :2] = np.where(opened, 0.04, 0.0)[:, None]
        return {"actions": native}
