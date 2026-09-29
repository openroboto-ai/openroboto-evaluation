"""Benchmark execution and scoring profiles used by the queue worker.

LIBERO-Pro registers flat runtime names such as ``libero_object_swap``, but
the score protocol keeps their semantic identity as ``(base_suite,
perturbation)``. A scoring profile can therefore change weights without
inventing another runtime suite or running an episode twice.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import pathlib
import re
import threading
from libero_eval.axis_runtime import (
    AXIS_V1_NAME,
    AXIS_V1_CONFIG_PATH,
    canonical_json_sha256,
    load_manifest,
)
from libero_eval.axis_release import AXIS_CURRENT_NAME, prepare_release


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
    expected_task_ids: tuple[int, ...] | None = None
    policy_seed: int | None = None
    protocol_revision: str | None = None
    manifest_sha256: str | None = None
    expected_trials_per_task: int | None = None
    manifest_path: pathlib.Path | None = None
    randomization_manifest_path: pathlib.Path | None = None
    randomization_manifest_sha256: str | None = None

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
ROBOTWIN_TARGETS = (ScoreTarget("robotwin_clean"),)
AXIS_VERSION_PATTERN = re.compile(r"axis_v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
DEFAULT_AXIS_DIRECTORY = AXIS_V1_CONFIG_PATH.parent


def is_axis_benchmark(name: str | None) -> bool:
    return isinstance(name, str) and AXIS_VERSION_PATTERN.fullmatch(name) is not None


def _axis_config_profile(name: str = AXIS_V1_NAME, path: pathlib.Path | None = None) -> EvaluationProfile:
    path = path or DEFAULT_AXIS_DIRECTORY / f"{name}.yaml"
    manifest = load_manifest(path, expected_name=name)
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict) or not isinstance(manifest.get("runtime"), dict):
        raise ValueError("AXIS configuration must declare runtime and protocol objects")
    trials = protocol.get("default_trials_per_task")
    if type(trials) is not int or trials < 1:
        raise ValueError("AXIS default_trials_per_task must be a positive integer")
    task_ids = tuple(task["task_id"] for task in manifest["tasks"])
    randomization_path = path.with_name("randomization.json") if manifest["protocol"].get("randomization") else None
    return EvaluationProfile(
        name,
        name if name == AXIS_V1_NAME else "axis",
        (ScoreTarget(name),),
        (1.0,),
        len(task_ids),
        task_ids,
        manifest["policy_seed"],
        manifest["protocol_revision"],
        canonical_json_sha256(manifest),
        trials,
        path.resolve(),
        randomization_path,
        canonical_json_sha256(json.loads(randomization_path.read_bytes())) if randomization_path else None,
    )


def _weights(targets: tuple[ScoreTarget, ...]) -> tuple[float, ...]:
    return (1.0,) * len(targets)


PROFILES = {
    AXIS_V1_NAME: _axis_config_profile(AXIS_V1_NAME),
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
    "robotwin": EvaluationProfile("robotwin", "robotwin", ROBOTWIN_TARGETS, (1.0,), 50),
}


_profile_lock = threading.RLock()
_axis_directory: pathlib.Path | None = None
_axis_dynamic: dict[str, EvaluationProfile] = {}
_axis_errors: dict[str, str] = {}
_axis_blocked: dict[str, str] = {}
_axis_verifications: dict[str, dict] = {}
logger = logging.getLogger("benchmark_worker")


class BenchmarkNotReadyError(ValueError):
    """A queue task must wait for a complete, validated local benchmark."""


def configure_axis_profiles(directory: pathlib.Path | None = None) -> None:
    """Configure once before the worker thread starts; built-in releases remain available."""
    global _axis_directory
    with _profile_lock:
        _axis_directory = directory.resolve() if directory is not None else None
        _axis_dynamic.clear()
        _axis_errors.clear()
        _axis_blocked.clear()
        _axis_verifications.clear()
    refresh_axis_profiles()


def block_axis_profile(name: str, reason: str | None) -> None:
    with _profile_lock:
        if reason is None:
            _axis_blocked.pop(name, None)
        else:
            _axis_blocked[name] = reason


def refresh_axis_profiles() -> None:
    """Admit complete, verified releases. Never replace the contents of a known version."""
    from tools.verify_axis_release import verify_release

    with _profile_lock:
        roots = [DEFAULT_AXIS_DIRECTORY]
        if _axis_directory is not None and _axis_directory != DEFAULT_AXIS_DIRECTORY:
            roots.append(_axis_directory)
        paths: dict[str, list[pathlib.Path]] = {}
        for root in roots:
            for path in sorted([*root.glob("axis_v*.yaml"), *root.glob("axis_v*/axis_v*.yaml")]):
                if AXIS_VERSION_PATTERN.fullmatch(path.stem):
                    paths.setdefault(path.stem, []).append(path.resolve())
        # Missing old bundles cannot silently fall back to an API or a different definition.
        for name in _axis_dynamic.keys() - paths.keys():
            _axis_errors[name] = "previously loaded AXIS version is missing from disk"
        for name, candidates in paths.items():
            try:
                profiles, verifications = [], []
                for path in candidates:
                    profile = _axis_config_profile(name, path)
                    manifest = load_manifest(path, expected_name=name)
                    snapshots = path.parent / manifest["task_snapshot_root"]
                    verification = verify_release(path, snapshots)
                    if "selector_sync" in manifest:
                        receipt = json.loads((path.parent / "selection.json").read_text(encoding="utf-8"))
                        if (
                            not isinstance(receipt, dict)
                            or receipt.get("configuration_verification") != verification
                            or receipt.get("selected_task_ids") != list(profile.expected_task_ids)
                        ):
                            raise ValueError("AXIS configuration differs from its frozen selection receipt")
                    verifications.append(verification)
                    profiles.append(profile)
                if len({p.manifest_sha256 for p in profiles}) != 1:
                    raise ValueError("conflicting definitions for the same AXIS version")
                profile, verification = profiles[0], verifications[0]
                old = _axis_dynamic.get(name) or PROFILES.get(name)
                if old is not None and old.manifest_sha256 != profile.manifest_sha256:
                    raise ValueError("AXIS version changed after loading; publish a new version instead")
                if name in _axis_verifications and _axis_verifications[name] != verification:
                    raise ValueError("frozen AXIS configuration or snapshots changed after loading")
                _axis_dynamic[name] = profile
                _axis_verifications[name] = verification
                _axis_errors.pop(name, None)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                message = str(exc)
                if _axis_errors.get(name) != message:
                    logger.warning("AXIS version %s not ready: %s", name, message)
                _axis_errors[name] = message


def get_profile(name: str) -> EvaluationProfile:
    with _profile_lock:
        if name == AXIS_CURRENT_NAME and name not in PROFILES:
            PROFILES[name] = _axis_config_profile(name, prepare_release())
        if is_axis_benchmark(name) and name not in _axis_dynamic:
            refresh_axis_profiles()
        if name in _axis_blocked or name in _axis_errors:
            raise BenchmarkNotReadyError(
                f"AXIS version {name} is not ready: {_axis_blocked.get(name) or _axis_errors[name]}"
            )
        profile = _axis_dynamic.get(name) or PROFILES.get(name)
        if profile is None:
            raise BenchmarkNotReadyError(
                f"unknown benchmark profile {name!r}; choose from {sorted(PROFILES | _axis_dynamic)}"
            )
        return profile


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
