"""Portable episode recordings for inspecting an Axis rollout after evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from typing import Any

import numpy as np
from PIL import Image


def save_recording(
    root: pathlib.Path,
    *,
    task_id: int,
    trial: int,
    frames: list[np.ndarray],
    steps: list[dict[str, Any]],
    control_period_s: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if not frames or len(frames) != len(steps) + 1:
        raise ValueError("recording requires a reset image plus one image per completed control step")
    if not math.isfinite(control_period_s) or control_period_s <= 0:
        raise ValueError("recording control period must be positive and finite")
    for frame in frames:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or frame.shape != frames[0].shape:
            raise ValueError("recording frames must have matching uint8 RGB shapes")
    name = f"axis_{task_id}_trial_{trial:03d}"
    animation, trace = root / f"{name}.gif", root / f"{name}.json"
    if animation.exists() or trace.exists():
        raise FileExistsError(f"episode recording already exists: {name}")
    encoded = (
        json.dumps(
            {
                "schema_version": 1,
                "task_id": task_id,
                "trial": trial,
                "control_period_s": control_period_s,
                "frame_count": len(frames),
                "metadata": metadata,
                "steps": steps,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    root.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    try:
        images[0].save(
            animation,
            save_all=True,
            append_images=images[1:],
            duration=round(control_period_s * 1000),
            loop=0,
            disposal=2,
        )
    finally:
        for image in images:
            image.close()
    trace.write_text(encoded, encoding="utf-8")
    return {
        "animation": f"{root.name}/{animation.name}",
        "trace": f"{root.name}/{trace.name}",
        "animation_sha256": hashlib.sha256(animation.read_bytes()).hexdigest(),
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "frame_count": len(frames),
        "control_period_s": control_period_s,
    }
