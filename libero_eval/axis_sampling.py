"""Reproducible sampling of tasks from a frozen AXIS pool (no model or GPU)."""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

from axis_runtime import canonical_json_sha256, task_specs


ALGORITHM = "axis-sha256-task-ranking-v1"


def sample_tasks(manifest: dict[str, Any], count: int, seed: int | None = None) -> dict[str, Any]:
    """Draw without replacement; preserve the seed and pool identity for replay.

    Each task receives a SHA-256 priority. Sorting priorities avoids depending
    on Python's random.sample implementation or the input task list order.
    The seed is local dev randomness, not a claim of a chain-derived lottery.
    """
    candidates = sorted(task_specs(manifest))
    if type(count) is not int or not 1 <= count <= len(candidates):
        raise ValueError(f"sample count must be an integer between 1 and {len(candidates)}")
    if seed is None:
        seed = secrets.randbits(64)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("sampling seed must be an integer in [0, 2**64)")

    def priority(task_id: int) -> tuple[bytes, int]:
        message = f"{ALGORITHM}:{seed}:{task_id}".encode("ascii")
        return hashlib.sha256(message).digest(), task_id

    return {
        "schema_version": 1,
        "algorithm": ALGORITHM,
        "benchmark": manifest["name"],
        "protocol_revision": manifest.get("protocol_revision"),
        "pool_manifest_sha256": canonical_json_sha256(manifest),
        "pool_task_ids": candidates,
        "pool_task_count": len(candidates),
        "sample_size": count,
        "sampling_seed": seed,
        "selected_task_ids": sorted(candidates, key=priority)[:count],
    }
