"""Queue seed and score-evidence checks for frozen AXIS randomization."""

import math
import re

from libero_eval.axis_randomization import load_randomization_plan, select_variant
from libero_eval.axis_runtime import load_manifest, task_specs

from benchmark_worker.profiles import BenchmarkNotReadyError


def queue_seed(task: dict) -> int:
    raw = task.get("seed")
    if isinstance(raw, str) and re.fullmatch(r"[0-9]+", raw):
        raw = int(raw)
    if type(raw) is not int or not 0 <= raw < 2**32:
        raise BenchmarkNotReadyError("randomized AXIS requires a queue-provided uint32 seed; no local fallback")
    return raw


def verify_summary(summary: dict, profile, seed: int) -> None:
    """Reject changed seeds/instances and inconsistent counts before score submission."""
    manifest = load_manifest(profile.manifest_path)
    plan = load_randomization_plan(
        profile.randomization_manifest_path,
        expected_benchmark=profile.name,
        expected_protocol_revision=profile.protocol_revision,
        benchmark_task_specs=task_specs(manifest),
    )
    if plan.manifest_canonical_sha256 != profile.randomization_manifest_sha256:
        raise ValueError("AXIS randomization manifest changed after profile loading")
    if (
        summary.get("randomization") is not True
        or type(summary.get("randomization_seed")) is not int
        or summary["randomization_seed"] != seed
    ):
        raise ValueError("AXIS summary does not use the queue randomization seed")
    if summary.get("score_reduction") != "task_mean":
        raise ValueError("AXIS randomized scoring requires task_mean")
    counts = []
    for task in summary["tasks"].values():
        episodes = task.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != task["num_trials"]:
            raise ValueError("AXIS summary is missing per-episode randomization evidence")
        for trial, episode in enumerate(episodes):
            expected = select_variant(plan, task_id=task["task_id"], trial=trial, seed=seed).provenance()
            if (
                not isinstance(episode, dict)
                or episode.get("trial") != trial
                or episode.get("randomization") != expected
                or episode.get("error") is not None
                or type(episode.get("success")) is not bool
            ):
                raise ValueError(f"AXIS task {task['task_id']} trial {trial} differs from the frozen plan")
        successes = sum(episode["success"] for episode in episodes)
        if task.get("num_successes") != successes or not math.isclose(task["success_rate"], successes / len(episodes)):
            raise ValueError("AXIS task success counts disagree with episode evidence")
        counts.append((successes, len(episodes)))
    total = sum(n for _, n in counts)
    successes = sum(n for n, _ in counts)
    score = sum(s / n for s, n in counts) / len(counts)
    suite = summary.get("suites", {}).get(profile.name, {})
    if (
        summary.get("total_episodes") != total
        or summary.get("total_successes") != successes
        or not math.isclose(summary.get("overall_success_rate", -1), score)
        or suite.get("episodes") != total
        or suite.get("successes") != successes
        or suite.get("tasks") != len(counts)
        or not math.isclose(suite.get("success_rate", -1), score)
    ):
        raise ValueError("AXIS total score disagrees with episode evidence")
