"""The public release works from its tracked bundle, including queue evidence checks."""

import copy
import json
import pathlib
import sys
import types
from unittest import mock

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "libero_eval"))

import axis_backend
from libero_eval.axis_release import prepare_release
from libero_eval.axis_randomization import load_randomization_plan, select_variant
from libero_eval.axis_runtime import load_manifest, task_specs
from benchmark_worker import worker
from benchmark_worker.profiles import BenchmarkNotReadyError, get_profile
from benchmark_worker.scoring import build_score_payload


def test_clean_extraction_and_corrupt_cache_fail_closed(tmp_path):
    path = prepare_release(cache_directory=tmp_path)
    assert load_manifest(path)["name"] == "axis_v2.0"
    assert len(list(path.parent.glob("payloads/*.json"))) == 600
    assert prepare_release(cache_directory=tmp_path) == path
    payload = next(path.parent.glob("payloads/*.json"))
    payload.write_text("{}")
    with pytest.raises(ValueError, match="differs from the published archive"):
        prepare_release(cache_directory=tmp_path)
    assert payload.read_text() == "{}"  # Never silently repair or replace evidence.


def test_archive_checksum_is_checked_before_cache_creation(tmp_path):
    receipt = json.loads((ROOT / "configs/benchmarks/axis_v2.0-release.json").read_bytes())
    (tmp_path / "axis_v2.0-release.json").write_text(json.dumps(receipt))
    (tmp_path / "axis_v2.0.tar.zst").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="archive checksum mismatch"):
        prepare_release(release_directory=tmp_path, cache_directory=tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


def test_named_cli_uses_published_randomization(monkeypatch):
    args = types.SimpleNamespace(
        benchmark="axis_v2.0", axis_manifest=None, axis_replan_steps=None, seed=None, config=None
    )
    assert args.benchmark == "axis_v2.0"
    axis_backend.apply_manifest_defaults(args)
    assert args.axis_loaded_manifest["protocol"]["randomization"] is True
    assert pathlib.Path(args.axis_randomization_manifest).name == "randomization.json"
    assert args.seed == 20260907
    args.axis_randomization_manifest = "/tmp/another-randomization.json"
    with pytest.raises(ValueError, match="cannot override"):
        axis_backend.apply_manifest_defaults(args)


def arguments():
    return types.SimpleNamespace(
        benchmark="axis_v2.0",
        eval_config=None,
        num_trials=20,
        gpus="4",
        workers_per_gpu=1,
        init_workers_per_gpu=1,
        server_impl="upstream",
        task_ids="",
        eval_timeout=60,
    )


@pytest.fixture(scope="module")
def complete_summary():
    profile = get_profile("axis_v2.0")
    manifest = load_manifest(profile.manifest_path)
    plan = load_randomization_plan(
        profile.randomization_manifest_path,
        expected_benchmark=profile.name,
        expected_protocol_revision=profile.protocol_revision,
        benchmark_task_specs=task_specs(manifest),
    )
    results = {}
    for task_id in profile.expected_task_ids:
        episodes = [
            {
                "trial": trial,
                "success": trial == 0,
                "error": None,
                "randomization": select_variant(plan, task_id=task_id, trial=trial, seed=20260928).provenance(),
            }
            for trial in range(20)
        ]
        results[task_id] = {
            "benchmark": profile.name,
            "task_id": task_id,
            "status": "ok",
            "num_trials": 20,
            "num_successes": 1,
            "success_rate": 0.05,
            "episodes": episodes,
        }
    return axis_backend._summary(
        results,
        {
            "benchmark": profile.name,
            "protocol_revision": profile.protocol_revision,
            "manifest_canonical_sha256": profile.manifest_sha256,
            "policy_seed": profile.policy_seed,
            "num_trials_per_task": 20,
            "dry_run": False,
            "randomization": True,
            "randomization_seed": 20260928,
            "score_reduction": "task_mean",
        },
    )


def test_queue_passes_seed_and_scores_all_instances(tmp_path, complete_summary):
    (tmp_path / "summary.json").write_text(json.dumps(complete_summary))
    task = {"benchmark": "axis_v2.0", "base_model": "pi0.5", "seed": "20260928"}
    worker.stop_event.clear()
    with mock.patch.object(worker.subprocess, "Popen", return_value=mock.Mock(returncode=0, poll=lambda: 0)) as launch:
        summary, error = worker.run_evaluation(task, tmp_path / "model", tmp_path, arguments())
    command = launch.call_args.args[0]
    assert command[command.index("--axis-randomization-seed") + 1] == "20260928"
    assert command[command.index("--axis-randomization-manifest") + 1].endswith("/randomization.json")
    assert command[command.index("--gpus") + 1] == "4"
    assert error == ""
    payload = build_score_payload(task, summary, 10, init_seed=20260928, benchmark="axis_v2.0")
    assert payload["success"] and payload["total_score"] == 0.05
    assert len(payload["per_task_scores"]) == 30 and payload["env_scores"][0]["samples"] == 600
    payload["init_seed"] = None
    assert "missing its queue seed" in worker.successful_payload_incomplete_reason(payload)


@pytest.mark.parametrize("seed", [None, True, -1, 2**32, 1.2, "1.2", "1_000", ""])
def test_invalid_queue_seed_rejected_before_launch(seed):
    with mock.patch.object(worker.subprocess, "Popen") as launch:
        with pytest.raises(BenchmarkNotReadyError, match="queue-provided uint32"):
            worker.task_evaluation_args({"seed": seed}, arguments())
        launch.assert_not_called()


@pytest.mark.parametrize("change", ["seed", "variant", "missing_episode", "task_count", "suite_score"])
def test_worker_rejects_wrong_randomization_evidence(tmp_path, complete_summary, change):
    summary = copy.deepcopy(complete_summary)
    task = next(iter(summary["tasks"].values()))
    if change == "seed":
        summary["randomization_seed"] += 1
    elif change == "variant":
        task["episodes"][0]["randomization"]["variant_id"] = "wrong"
    elif change == "missing_episode":
        task["episodes"].pop()
    elif change == "task_count":
        task["num_successes"] += 1
    else:
        summary["suites"]["axis_v2.0"]["success_rate"] = 1.0
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    worker.stop_event.clear()
    with mock.patch.object(worker.subprocess, "Popen", return_value=mock.Mock(returncode=0, poll=lambda: 0)):
        with pytest.raises(worker.EvalInfrastructureError, match="invalid AXIS randomization evidence"):
            worker.run_evaluation({"base_model": "pi0.5", "seed": 20260928}, tmp_path / "model", tmp_path, arguments())
