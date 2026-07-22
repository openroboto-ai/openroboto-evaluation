"""Prompt derivation for LIBERO-plus tasks (pure stdlib; unit-testable).

LIBERO-plus (arXiv:2510.13626) derives `task.language` from the bddl
*filename*, and for non-language perturbations the name carries encoded
parameters — `_view_0_0_100_0_0_initstate_3`, `_table_1`, `_tb_3`,
`_light_02`, `_add_10`, `_level1_sample1`, `_noise_38` — which appear in the
prompt as trailing text ("... place it on the plate view 0 0 100 0 0
initstate 3"). Issue sylvestf/LIBERO-plus#48 asks whether that is a bug, but
has no maintainer resolution. The published evaluator passes task.language
unchanged, so this helper is an ablation only and is not used by the official
benchmark profile.

`_language_` tasks are exempt: for those the fork parses the paraphrased
instruction from the bddl (:language) block, so task.language is already the
(correct) perturbed prompt — including combined names like
`..._language_10_view_...`, where it parses the `_language_10` bddl.
"""

import re

# A perturbation suffix starts at the first marker word followed by a number
# (`_table_1`, `_level1`); everything after it is parameters. Requiring the
# digit keeps instruction words like "..._on_the_table" or "..._light" safe.
_SUFFIX_RE = re.compile(r"_(?:view|table|tb|light|add|noise|level)_?-?\d.*$")


def clean_task_prompt(task_name: str, task_language: str) -> str:
    """Return the instruction for a LIBERO-plus task without filename junk."""
    if "_language_" in task_name:
        return str(task_language)
    base = _SUFFIX_RE.sub("", task_name)
    if base[:1].isupper():
        # LIBERO-100-style names ("KITCHEN_SCENE3_turn_on_the_stove_..."):
        # the instruction starts after the SCENE<i>_ prefix.
        base = base[base.find("SCENE") :]
        base = base[base.find("_") + 1 :]
    return " ".join(base.split("_"))
