"""Optional averaged inference weights, preserving every frozen parameter exactly."""

from __future__ import annotations

import math


def validate_ema_decay(value: float | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value < 1
    ):
        raise ValueError("ema_decay must be None or a finite number strictly between 0 and 1")


def inference_parameters(params, ema_params, trainable_filter):
    if ema_params is None:
        return params

    from flax import nnx

    # Upstream EMA also performs arithmetic on frozen weights. Even old == new
    # can round after multiply/add, so export their unchanged live values.
    return nnx.State.merge(params.filter(nnx.Not(trainable_filter)), ema_params.filter(trainable_filter))
