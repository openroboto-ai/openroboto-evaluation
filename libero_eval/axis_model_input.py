"""AXIS inference contract and historical training-metadata interpretation."""

from __future__ import annotations

# Standard Pi0.5 encodes state in prompt tokens. The LIBERO-specific false
# override is not the AXIS submission contract; submitted files cannot change it.
AXIS_PI05_DISCRETE_STATE_INPUT = True


def model_uses_discrete_state(metadata: dict) -> bool:
    """Read historical training provenance, never select evaluation inputs."""
    if not isinstance(metadata, dict):
        raise ValueError("AXIS checkpoint metadata must be an object")
    value = metadata.get("discrete_state_input", False)
    if type(value) is not bool:
        raise ValueError("AXIS checkpoint discrete_state_input must be a boolean")
    return value
