from __future__ import annotations
from typing import Any
import numpy as np

def _jitter_material(
    random_state: np.random.RandomState,
    material: tuple[float, float, float],
    amount: float = 0.08,
) -> np.ndarray:
    return np.clip(np.asarray(material, dtype=np.float64) + random_state.uniform(-amount, amount, size=3), 0.0, 1.0)


def _apply_image_luminance_guard(
    bitmap: np.ndarray,
    guard: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if bitmap.ndim != 3 or bitmap.shape[-1] not in (3, 4):
        raise ValueError(f"Image luminance guard requires RGB or RGBA bitmap, got {bitmap.shape}.")
    if bitmap.dtype != np.uint8:
        raise ValueError(f"Image luminance guard requires uint8 bitmap, got {bitmap.dtype}.")
    percentile = float(guard["percentile"])
    maximum = float(guard["max_value"])
    rgb = np.asarray(bitmap[..., :3], dtype=np.float64)
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    source_luminance = rgb @ weights
    source_value = float(np.percentile(source_luminance, percentile))
    gain = min(1.0, maximum / source_value) if source_value > 0.0 else 1.0
    resolved = np.asarray(bitmap).copy()
    if gain < 1.0:
        resolved[..., :3] = np.clip(np.floor(rgb * gain), 0.0, 255.0).astype(np.uint8)
    resolved_luminance = np.asarray(resolved[..., :3], dtype=np.float64) @ weights
    resolved_value = float(np.percentile(resolved_luminance, percentile))
    if resolved_value > maximum + 1e-9:
        raise ValueError(
            "Downscale-only image luminance guard failed its configured limit: "
            f"resolved={resolved_value}, maximum={maximum}."
        )
    return resolved, {
        "metric": guard["metric"],
        "mode": guard["mode"],
        "percentile": percentile,
        "max_value": maximum,
        "source_percentile_value": source_value,
        "resolved_percentile_value": resolved_value,
        "gain": float(gain),
        "applied": bool(gain < 1.0),
    }
