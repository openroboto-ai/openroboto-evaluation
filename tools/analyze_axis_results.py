#!/usr/bin/env python3
"""Analyze complete evaluation summaries against their frozen Axis manifest."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import statistics
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))
from axis_runtime import DEFAULT_MANIFEST, load_manifest


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(sum(value * value for value in left_delta) * sum(value * value for value in right_delta))
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta, strict=True)) / denominator


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [0.0, 0.0]
    rate = successes / trials
    scale = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / scale
    half_width = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / scale
    return [round(max(0.0, center - half_width), 6), round(min(1.0, center + half_width), 6)]


def _failed_conditions(checker: dict[str, Any]) -> set[str]:
    if checker.get("passed") is True:
        return set()
    children = checker.get("sub_results")
    if isinstance(children, list):
        return set().union(*(_failed_conditions(child) for child in children if isinstance(child, dict)))
    if checker.get("passed") is False:
        return {str(checker.get("checker_type", "unknown-checker"))}
    return set()


def _parse_summary(label: str, summary: dict[str, Any], manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    expected_ids = {int(task["task_id"]) for task in manifest["tasks"]}
    if summary.get("benchmark") != manifest["name"]:
        raise ValueError(f"{label}: expected benchmark={manifest['name']!r}")
    if summary.get("dry_run"):
        raise ValueError(f"{label}: dry-run/environment-smoke summaries are not model results")
    if summary.get("randomization") is not manifest["protocol"]["randomization"]:
        raise ValueError(f"{label}: randomization does not match the manifest")
    if manifest.get("protocol_revision") and summary.get("protocol_revision") != manifest["protocol_revision"]:
        raise ValueError(f"{label}: protocol_revision does not match the manifest")
    if summary.get("manifest_canonical_sha256") is not None:
        digest = hashlib.sha256(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        if summary["manifest_canonical_sha256"] != digest:
            raise ValueError(f"{label}: manifest hash does not match")
    tasks = summary.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"{label}: summary.tasks must be an object")

    parsed: dict[int, dict[str, Any]] = {}
    for raw in tasks.values():
        if not isinstance(raw, dict) or raw.get("status") != "ok":
            raise ValueError(f"{label}: every frozen task must have status='ok'")
        task_id = int(raw["task_id"])
        trials = int(raw["num_trials"])
        successes = int(raw["num_successes"])
        rate = float(raw["success_rate"])
        if task_id in parsed or trials <= 0 or not 0 <= successes <= trials:
            raise ValueError(f"{label}: invalid task record for {task_id}")
        if not math.isfinite(rate) or not math.isclose(rate, successes / trials, abs_tol=1e-9):
            raise ValueError(f"{label}: inconsistent success rate for task {task_id}")
        conditions: collections.Counter[str] = collections.Counter()
        episodes = raw.get("episodes")
        if episodes is not None:
            if not isinstance(episodes, list) or len(episodes) != trials:
                raise ValueError(f"{label}: episode count does not match task {task_id}")
            for episode in episodes:
                if not isinstance(episode, dict) or not isinstance(episode.get("success"), bool):
                    raise ValueError(f"{label}: invalid episode in task {task_id}")
                if episode.get("error") is not None:
                    raise ValueError(f"{label}: infrastructure error in task {task_id}; cannot treat as model failure")
                if not episode["success"]:
                    conditions.update(_failed_conditions(episode.get("checker") or {}))
            if sum(episode["success"] for episode in episodes) != successes:
                raise ValueError(f"{label}: episode successes do not match task {task_id}")
        parsed[task_id] = {
            "successes": successes,
            "trials": trials,
            "success_rate": rate,
            "goal_not_reached": trials - successes,
            "failed_terminal_conditions": dict(conditions),
            "episode_evidence_available": episodes is not None,
        }
    if set(parsed) != expected_ids:
        missing = sorted(expected_ids - set(parsed))
        extra = sorted(set(parsed) - expected_ids)
        raise ValueError(f"{label}: task ids do not match frozen manifest; missing={missing} extra={extra}")
    return parsed


def _common_recorded_protocol(named_summaries: list[tuple[str, dict[str, Any]]]) -> dict[str, Any] | None:
    """Compare effective settings, which may override a shared manifest's defaults.

    Legacy single-run descriptive statistics remain useful without these fields.
    Matching summary fields does not certify evaluator source or hardware parity;
    those still require the original run receipts and environment evidence.
    """
    if len(named_summaries) < 2:
        return None
    fields = [
        "policy_seed",
        "num_trials_per_task",
        "replan_steps",
        "gripper_mode",
        "policy_samples",
        "sample_reduction",
        "renderer_backend",
    ]
    if named_summaries[0][1]["randomization"]:
        fields.append("randomization_seed")
    baseline_label, baseline = named_summaries[0]
    for label, summary in named_summaries:
        for field in fields:
            if summary.get(field) is None:
                raise ValueError(
                    f"{label}: multi-model statistics require explicit {field}; "
                    "use complete evaluator summaries or analyze this run separately"
                )
            if type(summary[field]) is not type(baseline.get(field)) or summary[field] != baseline[field]:
                raise ValueError(
                    f"{label}: multi-model statistics require matching {field}; "
                    f"got {summary[field]!r}, but {baseline_label} records {baseline[field]!r}"
                )
    return {field: baseline[field] for field in fields}


def analyze(
    named_summaries: list[tuple[str, dict[str, Any]]],
    manifest: dict[str, Any],
    *,
    floor: float = 0.05,
    ceiling: float = 0.95,
    min_separation: float = 0.2,
    redundancy: float = 0.9,
) -> dict[str, Any]:
    specs = {int(task["task_id"]): task for task in manifest["tasks"]}
    if not named_summaries:
        raise ValueError("at least one model summary is required")
    if len({label for label, _ in named_summaries}) != len(named_summaries):
        raise ValueError("run labels must be unique")
    runs = {label: _parse_summary(label, summary, manifest) for label, summary in named_summaries}
    common_protocol = _common_recorded_protocol(named_summaries)
    labels = list(runs)

    model_statistics = []
    for label in labels:
        successes = sum(row["successes"] for row in runs[label].values())
        trials = sum(row["trials"] for row in runs[label].values())
        model_statistics.append({
            "model": label,
            "successes": successes,
            "trials": trials,
            "success_rate": round(successes / trials, 6),
            "wilson_95": _wilson(successes, trials),
        })

    task_statistics = []
    vectors: dict[int, list[float]] = {}
    for task_id, spec in specs.items():
        rates = [runs[label][task_id]["success_rate"] for label in labels]
        vectors[task_id] = rates
        successes = sum(runs[label][task_id]["successes"] for label in labels)
        trials = sum(runs[label][task_id]["trials"] for label in labels)
        task_statistics.append({
            "task_id": task_id,
            "instruction": spec["instruction"],
            "capability": spec["skill"],
            "model_success_rates": dict(zip(labels, rates, strict=True)),
            "mean_model_success_rate": round(statistics.fmean(rates), 6),
            "pooled_success_rate": round(successes / trials, 6),
            "wilson_95": _wilson(successes, trials),
            "model_separation": round(max(rates) - min(rates), 6),
            "floor_effect": max(rates) <= floor,
            "ceiling_effect": min(rates) >= ceiling,
            "model_failures": {
                label: {
                    key: runs[label][task_id][key]
                    for key in ("goal_not_reached", "failed_terminal_conditions", "episode_evidence_available")
                }
                for label in labels
            },
        })

    capabilities = sorted({spec["skill"] for spec in specs.values()})
    capability_statistics = []
    for capability in capabilities:
        task_ids = [task_id for task_id, spec in specs.items() if spec["skill"] == capability]
        per_model = {
            label: round(statistics.fmean(runs[label][task_id]["success_rate"] for task_id in task_ids), 6)
            for label in labels
        }
        capability_statistics.append({
            "capability": capability,
            "task_ids": task_ids,
            "model_success_rates": per_model,
            "model_separation": round(max(per_model.values()) - min(per_model.values()), 6),
        })

    correlations = []
    for left_index, left in enumerate(specs):
        for right in list(specs)[left_index + 1 :]:
            pearson = _pearson(vectors[left], vectors[right])
            spearman = _pearson(_ranks(vectors[left]), _ranks(vectors[right])) if len(labels) >= 3 else None
            correlations.append({
                "left_task_id": left,
                "right_task_id": right,
                "pearson": None if pearson is None else round(pearson, 6),
                "spearman": None if spearman is None else round(spearman, 6),
            })

    warnings = []
    recommended: list[int] = []
    if len(labels) < 3:
        warnings.append(
            "At least three protocol-compatible model runs are required for correlation and subset selection."
        )
    else:
        candidates = [
            row
            for row in task_statistics
            if not row["floor_effect"] and not row["ceiling_effect"] and row["model_separation"] >= min_separation
        ]
        candidates.sort(key=lambda row: (-row["model_separation"], row["task_id"]))
        correlation_map = {
            frozenset((row["left_task_id"], row["right_task_id"])): row["spearman"] for row in correlations
        }
        for candidate in candidates:
            task_id = candidate["task_id"]
            if any(
                (correlation_map.get(frozenset((task_id, selected))) is not None)
                and abs(correlation_map[frozenset((task_id, selected))]) >= redundancy
                for selected in recommended
            ):
                continue
            recommended.append(task_id)
        if not recommended:
            warnings.append("No task met the frozen separation/floor/ceiling thresholds; no subset is recommended.")

    return {
        "schema_version": 1,
        "benchmark": manifest["name"],
        "protocol_revision": manifest.get("protocol_revision"),
        "randomization": manifest["protocol"]["randomization"],
        "failure_interpretation": (
            "Final checker conditions describe the failed goal; they do not identify causal failure modes."
        ),
        "model_count": len(labels),
        "common_recorded_protocol": common_protocol,
        "thresholds": {
            "floor": floor,
            "ceiling": ceiling,
            "min_model_separation": min_separation,
            "redundancy_abs_spearman": redundancy,
        },
        "model_statistics": model_statistics,
        "task_statistics": task_statistics,
        "capability_statistics": capability_statistics,
        "task_correlations": correlations,
        "recommended_task_ids": recommended,
        "warnings": warnings,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['benchmark']} model/task statistics",
        "",
        f"Models: {report['model_count']}; randomization: `{str(report['randomization']).lower()}`.",
        "",
        "## Model results",
        "",
        "| Model | Successes/trials | Success rate | Wilson 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for row in report["model_statistics"]:
        lines.append(
            f"| {row['model']} | {row['successes']}/{row['trials']} | {row['success_rate']:.3f} | "
            f"[{row['wilson_95'][0]:.3f}, {row['wilson_95'][1]:.3f}] |"
        )
    lines += [
        "",
        "## Task statistics",
        "",
        "| Task | Capability | Mean SR | Separation | Floor | Ceiling |",
        "|---:|---|---:|---:|---|---|",
    ]
    for row in report["task_statistics"]:
        lines.append(
            f"| {row['task_id']} | {row['capability']} | {row['mean_model_success_rate']:.3f} | "
            f"{row['model_separation']:.3f} | {row['floor_effect']} | {row['ceiling_effect']} |"
        )
    lines += ["", "## Recommended subset", "", ", ".join(map(str, report["recommended_task_ids"])) or "None."]
    lines += [
        "",
        "## Failed terminal conditions",
        "",
        "These are checker outcomes, not diagnoses such as missed grasp or dropped object.",
        "",
        "| Task | Model | Goal not reached | Failed conditions |",
        "|---:|---|---:|---|",
    ]
    for row in report["task_statistics"]:
        for label, failure in row["model_failures"].items():
            lines.append(
                f"| {row['task_id']} | {label} | {failure['goal_not_reached']} | "
                f"{json.dumps(failure['failed_terminal_conditions'], sort_keys=True)} |"
            )
    if report["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {warning}" for warning in report["warnings"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=SUMMARY_JSON")
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--markdown", type=pathlib.Path)
    args = parser.parse_args()

    named_summaries = []
    for value in args.run:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            parser.error(f"--run must be LABEL=SUMMARY_JSON, got {value!r}")
        path = pathlib.Path(raw_path).expanduser().resolve()
        named_summaries.append((label, json.loads(path.read_text(encoding="utf-8"))))
    manifest = load_manifest(args.manifest)
    report = analyze(named_summaries, manifest)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
