"""Parameter selection for the pinned Pi0.5 action expert and its projections."""

from typing import Any


TRAINING_SCOPES = ("full", "action-expert")
ACTION_PROJECTIONS = frozenset({
    "action_in_proj",
    "action_out_proj",
    "time_mlp_in",
    "time_mlp_out",
    "state_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
})


def action_expert_parameter(path: tuple, value: Any) -> bool:
    """Select NNX parameter paths, before Orbax adds its outer `params` key."""
    del value
    if not path:
        return False
    if path[0] in ACTION_PROJECTIONS:
        return True
    # The pinned OpenPI Gemma module names the second expert's modules with _1.
    # Restrict that convention to the language module, excluding vision layers.
    return path[:2] == ("PaliGemma", "llm") and any(isinstance(part, str) and part.endswith("_1") for part in path[2:])
