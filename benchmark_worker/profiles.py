"""Benchmark execution and scoring profiles used by the queue worker.

LIBERO-Pro registers flat runtime names such as ``libero_object_swap``, but
the score protocol keeps their semantic identity as ``(base_suite,
perturbation)``. A scoring profile can therefore change weights without
inventing another runtime suite or running an episode twice.
"""

from __future__ import annotations

from dataclasses import dataclass


PRO_BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PRO_PERTURBATIONS = ("object", "swap", "lan", "task")
BASE_SUITES = PRO_BASE_SUITES


@dataclass(frozen=True)
class ScoreTarget:
    base_suite: str
    perturbation: str | None = None

    @property
    def env_name(self) -> str:
        return self.base_suite if self.perturbation is None else f"{self.base_suite}_{self.perturbation}"


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    runtime_benchmark: str
    targets: tuple[ScoreTarget, ...]
    weights: tuple[float, ...]
    expected_task_count: int

    def weight_for(self, target: ScoreTarget) -> float:
        return self.weights[self.targets.index(target)]


BASE_TARGETS = tuple(ScoreTarget(suite) for suite in BASE_SUITES)
PRO_TARGETS = tuple(ScoreTarget(suite, perturbation) for suite in PRO_BASE_SUITES for perturbation in PRO_PERTURBATIONS)

# LIBERO-Plus leaderboard Total is a micro-average over all 10,030 task
# variants.  Weighting the four suite success rates by their registry sizes is
# exactly equivalent while preserving the existing score payload schema.
PLUS_SUITE_TASK_COUNTS = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}
PLUS_WEIGHTS = tuple(float(PLUS_SUITE_TASK_COUNTS[target.base_suite]) for target in BASE_TARGETS)


def _weights(targets: tuple[ScoreTarget, ...]) -> tuple[float, ...]:
    return (1.0,) * len(targets)


PROFILES = {
    "libero": EvaluationProfile("libero", "libero", BASE_TARGETS, _weights(BASE_TARGETS), 40),
    "libero_pro": EvaluationProfile("libero_pro", "libero_pro", PRO_TARGETS, _weights(PRO_TARGETS), 160),
    "libero_pro_custom_1": EvaluationProfile(
        "libero_pro_custom_1",
        "libero_pro",
        PRO_TARGETS,
        # Custom 1 的差异在评分协议边界折叠成六条记录；运行层仍是等权的
        # 16 个 LIBERO-Pro suite，不在这里重复表达上报权重。
        _weights(PRO_TARGETS),
        160,
    ),
    "libero_plus": EvaluationProfile("libero_plus", "libero_plus", BASE_TARGETS, PLUS_WEIGHTS, 10030),
}


def get_profile(name: str) -> EvaluationProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark profile {name!r}; choose from {sorted(PROFILES)}") from exc


def target_from_env_name(env_name: str) -> ScoreTarget:
    """Parse a runtime suite name without confusing a suffix with a base."""
    if env_name in BASE_SUITES:
        return ScoreTarget(env_name)
    for base_suite in sorted(PRO_BASE_SUITES, key=len, reverse=True):
        prefix = f"{base_suite}_"
        if env_name.startswith(prefix):
            perturbation = env_name[len(prefix) :]
            if perturbation in PRO_PERTURBATIONS:
                return ScoreTarget(base_suite, perturbation)
    raise ValueError(f"unknown benchmark suite {env_name!r}")
