"""Policy-backbone registry shared by the evaluator CLIs.

Backbones describe the model/runtime contract.  Benchmarks describe the task
and simulator contract.  Keeping those axes separate is what lets a future
checkpoint be evaluated on another compatible benchmark without adding a new
cross-product of command-line flags.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Backbone:
    name: str
    model_family: str
    legacy_architectures: tuple[str, ...] = ()


BACKBONES = {
    "pi0": Backbone("pi0", "openpi", ("pi0",)),
    "pi0.5": Backbone("pi0.5", "openpi", ("pi0.5",)),
    "openvla-oft": Backbone("openvla-oft", "openvla_oft"),
    "lingbot-vla-v2": Backbone("lingbot-vla-v2", "lingbot_vla_v2"),
}

_ALIASES = {
    "pi0": "pi0",
    "π0": "pi0",
    "pi0.5": "pi0.5",
    "pi05": "pi0.5",
    "π0.5": "pi0.5",
    "openvla_oft": "openvla-oft",
    "openvla-oft": "openvla-oft",
    "lingbot_vla_v2": "lingbot-vla-v2",
    "lingbot-vla-v2": "lingbot-vla-v2",
    "lingbot-vla-2.0": "lingbot-vla-v2",
}


def parse_backbone(value: str) -> Backbone:
    token = value.strip().lower()
    try:
        return BACKBONES[_ALIASES[token]]
    except KeyError as exc:
        raise ValueError(f"unknown backbone {value!r}; choose from {', '.join(BACKBONES)}") from exc


def resolve_backbone(
    backbone: str | None,
    legacy_architectures: tuple[str, ...] | None = None,
    requested_family: str = "auto",
) -> tuple[Backbone, tuple[str, ...]]:
    """Resolve the new explicit flag while preserving the old architecture flag.

    ``--model-architectures`` used to be an allow-list.  It remains supported
    for old commands, including the multi-architecture development form.  New
    production commands should select exactly one ``--backbone``.
    """
    if backbone is not None:
        selected = parse_backbone(backbone)
        if legacy_architectures:
            if not selected.legacy_architectures or tuple(legacy_architectures) != selected.legacy_architectures:
                raise ValueError(
                    f"--backbone {selected.name!r} conflicts with legacy --model-architectures "
                    f"{','.join(legacy_architectures)!r}; omit --model-architectures"
                )
        architectures = selected.legacy_architectures
    elif legacy_architectures:
        # A multi-value legacy allow-list has no single model identity before
        # checkpoint inspection.  The openpi runtime remains its shared family.
        selected = BACKBONES[legacy_architectures[0]]
        architectures = tuple(legacy_architectures)
    elif requested_family == "openvla_oft":
        selected = BACKBONES["openvla-oft"]
        architectures = ()
    elif requested_family == "lingbot_vla_v2":
        selected = BACKBONES["lingbot-vla-v2"]
        architectures = ()
    else:
        # Exact historical behavior when no new flag is supplied.
        selected = BACKBONES["pi0.5"]
        architectures = selected.legacy_architectures

    if requested_family != "auto" and requested_family != selected.model_family:
        raise ValueError(
            f"--backbone {selected.name!r} uses model family {selected.model_family!r}, "
            f"not requested --model-family {requested_family!r}"
        )
    return selected, architectures
