"""Deterministic request-level RNG protocol for LingBot-VLA 2.0 evaluation."""

from __future__ import annotations

import hashlib


POLICY_RNG_FIELD = "_evaluation_seed"
POLICY_RNG_MODE = "per_inference_sha256_v1"
POLICY_BATCH_LANE_FIELD = "_evaluation_batch_lane"
POLICY_BATCH_LANE_RELEASE_FIELD = "_evaluation_batch_lane_release"
POLICY_BATCH_MODE_STATIC = "static_lane_batch_v1"
_MAX_POLICY_SEED = (1 << 63) - 1


def derive_policy_seed(base_seed: int, suite: str, task_id: int, trial: int, infer_call: int) -> int:
    """Derive a stable per-inference seed independently of worker scheduling."""
    integer_fields = {
        "base_seed": base_seed,
        "task_id": task_id,
        "trial": trial,
        "infer_call": infer_call,
    }
    for name, value in integer_fields.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if not isinstance(suite, str) or not suite:
        raise ValueError(f"suite must be a non-empty string, got {suite!r}")

    payload = (f"lingbot-libero-policy-rng-v1\0{base_seed}\0{suite}\0{task_id}\0{trial}\0{infer_call}").encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _MAX_POLICY_SEED


def validate_policy_seed(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= _MAX_POLICY_SEED:
        raise ValueError(f"evaluation seed must be an integer in [0, {_MAX_POLICY_SEED}], got {value!r}")
    return value


def validate_policy_batch_lane(value: object) -> int:
    """Validate the stable evaluator lane carried by static-batch requests."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"evaluation batch lane must be a non-negative integer, got {value!r}")
    return value
