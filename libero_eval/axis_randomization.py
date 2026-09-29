"""Versioned, deterministic AXIS scene-variant selection.

The framework consumes a release-owned manifest whose payloads and semantic
fields are all SHA-256 pinned. It deliberately has no online fallback: a
randomized benchmark must remain reproducible after the upstream API changes.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any

if __package__:
    from .axis_runtime import canonical_json_sha256, verify_task_payload
else:
    from axis_runtime import canonical_json_sha256, verify_task_payload


ALGORITHM = "sha256-rejection-v1"
PERMUTATION_ALGORITHM = "sha256-cycle-permutation-v1"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_TOP_LEVEL_KEYS = {"schema_version", "benchmark", "protocol_revision", "seed_contract", "tasks"}
_SEED_CONTRACT_KEYS = {"source", "algorithm", "namespace"}
_TASK_KEYS = {"task_id", "instruction", "variants"}
_VARIANT_KEYS = {
    "variant_id",
    "payload_path",
    "payload_canonical_sha256",
    "mjcf_sha256",
    "checker_sha256",
    "initial_state_sha256",
    "dimensions",
}


@dataclass(frozen=True)
class VariantSpec:
    task_id: int
    instruction: str
    variant_id: str
    payload_path: pathlib.Path
    payload_canonical_sha256: str
    mjcf_sha256: str
    checker_sha256: str
    initial_state_sha256: str
    dimensions: dict[str, Any]
    official_randomization_sha256: str | None = None


@dataclass(frozen=True)
class RandomizationPlan:
    benchmark: str
    protocol_revision: str
    namespace: str
    manifest_path: pathlib.Path
    manifest_canonical_sha256: str
    variants_by_task: dict[int, tuple[VariantSpec, ...]]
    enabled_by_task: dict[int, bool] = field(default_factory=dict)
    algorithm: str = ALGORITHM


@dataclass(frozen=True)
class VariantSelection:
    spec: VariantSpec
    seed: int
    trial: int
    variant_index: int
    variant_count: int
    selection_digest: str
    randomization_manifest_sha256: str
    enabled: bool = True
    algorithm: str = ALGORITHM

    def provenance(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "trial": self.trial,
            "variant_id": self.spec.variant_id,
            "variant_index": self.variant_index,
            "variant_count": self.variant_count,
            "selection_digest": self.selection_digest,
            "randomization_manifest_sha256": self.randomization_manifest_sha256,
            "payload_canonical_sha256": self.spec.payload_canonical_sha256,
            "dimensions": self.spec.dimensions,
            **(
                {"official_randomization_sha256": self.spec.official_randomization_sha256}
                if self.spec.official_randomization_sha256 is not None
                else {}
            ),
        }


@dataclass(frozen=True)
class ResolvedVariant:
    selection: VariantSelection
    payload: dict[str, Any]


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _safe_payload_path(root: pathlib.Path, value: Any, field: str) -> pathlib.Path:
    if not isinstance(value, str) or not value or "\\" in value or pathlib.PurePosixPath(value).is_absolute():
        raise ValueError(f"{field} must be a non-empty relative POSIX path")
    pure = pathlib.PurePosixPath(value)
    if ".." in pure.parts or pure.suffix != ".json":
        raise ValueError(f"{field} must stay inside the manifest directory and end in .json")
    resolved = (root / pathlib.Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes the manifest directory") from exc
    return resolved


def _reject_constant(value: str) -> None:
    raise ValueError(f"AXIS randomization JSON contains non-finite number {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"AXIS randomization JSON repeats key {key!r}")
        result[key] = value
    return result


def _load_strict_json(path: pathlib.Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )


def _validate_dimensions(value: Any, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_dimensions(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} contains a non-string key")
            _validate_dimensions(item, f"{field}.{key}")
        return
    raise ValueError(f"{field} contains unsupported JSON value {type(value).__name__}")


def load_randomization_plan(
    path: pathlib.Path,
    *,
    expected_benchmark: str,
    expected_protocol_revision: str,
    benchmark_task_specs: dict[int, dict[str, Any]],
) -> RandomizationPlan:
    path = path.expanduser().resolve()
    raw = _load_strict_json(path)
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw["schema_version"] not in (1, 2):
        raise ValueError("AXIS randomization manifest must use schema_version=1 or 2")
    if set(raw) != _TOP_LEVEL_KEYS:
        raise ValueError(f"AXIS randomization manifest fields must be exactly {sorted(_TOP_LEVEL_KEYS)}")
    if raw.get("benchmark") != expected_benchmark:
        raise ValueError(
            f"AXIS randomization benchmark mismatch: got {raw.get('benchmark')!r}, expected {expected_benchmark!r}"
        )
    if raw.get("protocol_revision") != expected_protocol_revision:
        raise ValueError(
            "AXIS randomization protocol revision mismatch: "
            f"got {raw.get('protocol_revision')!r}, expected {expected_protocol_revision!r}"
        )
    seed_contract = raw.get("seed_contract")
    if not isinstance(seed_contract, dict):
        raise ValueError("AXIS randomization manifest must define seed_contract")
    if set(seed_contract) != _SEED_CONTRACT_KEYS:
        raise ValueError(f"AXIS randomization seed_contract fields must be exactly {sorted(_SEED_CONTRACT_KEYS)}")
    allowed_algorithms = {ALGORITHM} if raw["schema_version"] == 1 else {ALGORITHM, PERMUTATION_ALGORITHM}
    if seed_contract.get("source") != "queue" or seed_contract.get("algorithm") not in allowed_algorithms:
        raise ValueError(f"AXIS randomization seed contract must be queue/{ALGORITHM}")
    namespace = seed_contract.get("namespace")
    if not isinstance(namespace, str) or _IDENTIFIER.fullmatch(namespace) is None:
        raise ValueError("AXIS randomization seed namespace must be a stable lowercase identifier")

    task_entries = raw.get("tasks")
    if not isinstance(task_entries, list):
        raise ValueError("AXIS randomization manifest tasks must be a list")
    entries_by_id: dict[int, dict[str, Any]] = {}
    for entry in task_entries:
        task_keys = _TASK_KEYS | ({"randomization_enabled"} if raw["schema_version"] == 2 else set())
        if not isinstance(entry, dict) or set(entry) != task_keys:
            raise ValueError(f"AXIS randomization task fields must be exactly {sorted(_TASK_KEYS)}")
        if isinstance(entry.get("task_id"), bool) or not isinstance(entry.get("task_id"), int):
            raise ValueError("AXIS randomization task entry must contain an integer task_id")
        task_id = entry["task_id"]
        if task_id in entries_by_id:
            raise ValueError(f"duplicate AXIS randomization task {task_id}")
        entries_by_id[task_id] = entry
    expected_ids = set(benchmark_task_specs)
    if set(entries_by_id) != expected_ids:
        raise ValueError(
            f"AXIS randomization task coverage mismatch: got {sorted(entries_by_id)}, expected {sorted(expected_ids)}"
        )

    variants_by_task: dict[int, tuple[VariantSpec, ...]] = {}
    enabled_by_task = {}
    for task_id, benchmark_spec in sorted(benchmark_task_specs.items()):
        entry = entries_by_id[task_id]
        if entry.get("instruction") != benchmark_spec["instruction"]:
            raise ValueError(f"AXIS randomization task {task_id} instruction mismatch")
        variants = entry.get("variants")
        enabled = entry.get("randomization_enabled", True)
        if type(enabled) is not bool:
            raise ValueError("randomization_enabled must be a boolean")
        enabled_by_task[task_id] = enabled
        if "randomization_enabled" in benchmark_spec and benchmark_spec["randomization_enabled"] != enabled:
            raise ValueError(f"AXIS task {task_id} randomization flag differs from the benchmark")
        if not enabled and (not isinstance(variants, list) or len(variants) != 1):
            raise ValueError(f"AXIS fixed task {task_id} must freeze exactly one variant")
        if not isinstance(variants, list) or (enabled and len(variants) < 2):
            raise ValueError(f"AXIS randomized task {task_id} must freeze at least two variants")
        seen: set[str] = set()
        parsed: list[VariantSpec] = []
        for index, variant in enumerate(variants):
            if not isinstance(variant, dict) or set(variant) not in (
                _VARIANT_KEYS,
                _VARIANT_KEYS | {"official_randomization_sha256"},
            ):
                raise ValueError(f"AXIS task {task_id} variant {index} fields must be exactly {sorted(_VARIANT_KEYS)}")
            variant_id = variant.get("variant_id")
            if not isinstance(variant_id, str) or _IDENTIFIER.fullmatch(variant_id) is None:
                raise ValueError(f"AXIS task {task_id} variant_id must be a stable lowercase identifier")
            if variant_id in seen:
                raise ValueError(f"AXIS task {task_id} repeats variant_id {variant_id!r}")
            seen.add(variant_id)
            dimensions = variant.get("dimensions")
            if not isinstance(dimensions, dict) or not dimensions:
                raise ValueError(f"AXIS task {task_id} variant {variant_id!r} must describe randomization dimensions")
            if any(
                not isinstance(key, str) or _IDENTIFIER.fullmatch(key) is None or value is None
                for key, value in dimensions.items()
            ):
                raise ValueError(f"AXIS task {task_id} variant {variant_id!r} has invalid dimension metadata")
            _validate_dimensions(dimensions, f"task {task_id} variant {variant_id} dimensions")
            parsed.append(
                VariantSpec(
                    task_id=task_id,
                    instruction=benchmark_spec["instruction"],
                    variant_id=variant_id,
                    payload_path=_safe_payload_path(path.parent, variant.get("payload_path"), "payload_path"),
                    payload_canonical_sha256=_require_sha256(
                        variant.get("payload_canonical_sha256"), "payload_canonical_sha256"
                    ),
                    mjcf_sha256=_require_sha256(variant.get("mjcf_sha256"), "mjcf_sha256"),
                    checker_sha256=_require_sha256(variant.get("checker_sha256"), "checker_sha256"),
                    initial_state_sha256=_require_sha256(variant.get("initial_state_sha256"), "initial_state_sha256"),
                    dimensions=dimensions,
                    official_randomization_sha256=(
                        _require_sha256(variant["official_randomization_sha256"], "official_randomization_sha256")
                        if "official_randomization_sha256" in variant
                        else None
                    ),
                )
            )
        physical_fingerprints = {
            (
                variant.mjcf_sha256,
                variant.checker_sha256,
                variant.initial_state_sha256,
                variant.official_randomization_sha256,
            )
            for variant in parsed
        }
        if enabled and len(physical_fingerprints) < 2:
            raise ValueError(
                f"AXIS randomized task {task_id} variants do not differ in "
                "MJCF, checker, initial-state or official-config hashes"
            )
        variants_by_task[task_id] = tuple(sorted(parsed, key=lambda item: item.variant_id))

    return RandomizationPlan(
        benchmark=expected_benchmark,
        protocol_revision=expected_protocol_revision,
        namespace=namespace,
        manifest_path=path,
        manifest_canonical_sha256=canonical_json_sha256(raw),
        variants_by_task=variants_by_task,
        enabled_by_task=enabled_by_task,
        algorithm=seed_contract["algorithm"],
    )


def select_variant(plan: RandomizationPlan, *, task_id: int, trial: int, seed: int) -> VariantSelection:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("AXIS randomization seed must be an unsigned 64-bit integer")
    if isinstance(trial, bool) or not isinstance(trial, int) or trial < 0:
        raise ValueError("AXIS randomization trial must be a non-negative integer")
    variants = plan.variants_by_task.get(task_id)
    if variants is None:
        raise ValueError(f"AXIS randomization manifest has no task {task_id}")

    modulus = len(variants)
    if plan.algorithm == PERMUTATION_ALGORITHM:
        # One independently seeded permutation per cycle. All frozen instances
        # are exercised before any repeat; model identity never enters the seed.
        ranked = sorted(
            (
                hashlib.sha256(
                    json.dumps(
                        [plan.namespace, seed, task_id, trial // modulus, variant.variant_id], separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                index,
            )
            for index, variant in enumerate(variants)
        )
        digest, index = ranked[trial % modulus]
        return VariantSelection(
            spec=variants[index],
            seed=seed,
            trial=trial,
            variant_index=index,
            variant_count=modulus,
            selection_digest=digest,
            randomization_manifest_sha256=plan.manifest_canonical_sha256,
            enabled=plan.enabled_by_task.get(task_id, True),
            algorithm=plan.algorithm,
        )
    limit = 2**256 - (2**256 % modulus)
    counter = 0
    while True:
        preimage = json.dumps(
            [plan.namespace, seed, task_id, trial, counter],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(preimage).hexdigest()
        value = int(digest, 16)
        if value < limit:
            index = value % modulus
            return VariantSelection(
                spec=variants[index],
                seed=seed,
                trial=trial,
                variant_index=index,
                variant_count=modulus,
                selection_digest=digest,
                randomization_manifest_sha256=plan.manifest_canonical_sha256,
                enabled=plan.enabled_by_task.get(task_id, True),
            )
        counter += 1


def resolve_variant(selection: VariantSelection) -> ResolvedVariant:
    spec = selection.spec
    if not spec.payload_path.is_file():
        raise FileNotFoundError(f"frozen AXIS variant payload is missing: {spec.payload_path}")
    payload = _load_strict_json(spec.payload_path)
    verified = verify_task_payload(
        payload,
        {
            "task_id": spec.task_id,
            "instruction": spec.instruction,
            "mjcf_sha256": spec.mjcf_sha256,
            "checker_sha256": spec.checker_sha256,
            "initial_state_sha256": spec.initial_state_sha256,
            **(
                {"official_randomization_sha256": spec.official_randomization_sha256}
                if spec.official_randomization_sha256 is not None
                else {}
            ),
        },
    )
    actual = canonical_json_sha256(verified)
    if actual != spec.payload_canonical_sha256:
        raise ValueError(
            f"AXIS task {spec.task_id} variant {spec.variant_id!r} payload drifted: "
            f"got {actual}, expected {spec.payload_canonical_sha256}"
        )
    return ResolvedVariant(selection=selection, payload=verified)


def build_trial_plan(
    plan: RandomizationPlan,
    *,
    task_id: int,
    num_trials: int,
    seed: int,
) -> list[ResolvedVariant]:
    if isinstance(num_trials, bool) or not isinstance(num_trials, int) or num_trials < 1:
        raise ValueError("AXIS randomized evaluation requires at least one trial")
    if not plan.enabled_by_task.get(task_id, True) and num_trials != 1:
        raise ValueError(f"AXIS fixed task {task_id} must run exactly one trial")
    selections = [select_variant(plan, task_id=task_id, trial=trial, seed=seed) for trial in range(num_trials)]
    return [resolve_variant(selection) for selection in selections]
