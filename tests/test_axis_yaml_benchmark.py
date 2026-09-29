"""Editable task selection and the run_eval/worker contract for AXIS."""

import contextlib
import io
import json
import pathlib
import shutil
import sys
import types
from unittest import mock

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "libero_eval"))

import axis_backend
import run_eval
from axis_runtime import (
    AXIS_V1_CONFIG_PATH,
    canonical_json_sha256,
    load_manifest,
    verify_task_payload,
)
from benchmark_worker import profiles, worker
from benchmark_worker.scoring import build_score_payload

TASK_IDS = [
    22,
    31,
    33,
    34,
    35,
    37,
    40,
    41,
    42,
    43,
    44,
    46,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    501,
    502,
    503,
    504,
    505,
    506,
    514,
    757,
]


def test_all_thirty_requested_tasks_have_verified_local_scenes():
    manifest = load_manifest(AXIS_V1_CONFIG_PATH)
    assert manifest["name"] == "axis_v1.0"
    assert [task["task_id"] for task in manifest["tasks"]] == TASK_IDS
    for task in manifest["tasks"]:
        assert task["name_zh"]
        snapshot = AXIS_V1_CONFIG_PATH.parent / manifest["task_snapshot_root"] / f"{task['task_id']}.json"
        assert verify_task_payload(json.loads(snapshot.read_text()), task)["id"] == task["task_id"]
    # Same display names are distinct scenes/checkers and must not be deduplicated.
    assert sum(task["instruction"] == "Place Black Bowl on Top of Cabinet" for task in manifest["tasks"]) == 3
    assert load_manifest() == manifest


def test_v1_keeps_the_frozen_thirty_task_protocol():
    manifest = load_manifest(AXIS_V1_CONFIG_PATH)
    assert manifest["protocol_revision"] == "axis_v1.0_30tasks_native_joint_osmesa_v1"
    assert manifest["policy_seed"] == 20260907
    assert manifest["protocol"]["default_trials_per_task"] == 20
    assert manifest["protocol"]["max_control_steps_per_trial"] == 120
    assert axis_backend._manifest_path("axis_v1.0") == AXIS_V1_CONFIG_PATH


@pytest.fixture
def editable_config(tmp_path):
    config = yaml.safe_load(AXIS_V1_CONFIG_PATH.read_text())
    shutil.copyfile(AXIS_V1_CONFIG_PATH.parent / config["source_manifest"], tmp_path / config["source_manifest"])
    path = tmp_path / AXIS_V1_CONFIG_PATH.name
    return config, path


def test_yaml_edits_control_runtime_and_worker_task_selection(editable_config):
    config, path = editable_config
    config["tasks"] = [config["tasks"][-1], config["tasks"][0]]
    path.write_text(yaml.safe_dump(config))
    manifest = load_manifest(path)
    assert [task["task_id"] for task in manifest["tasks"]] == [757, 22]
    profile = profiles._axis_config_profile(path=path)
    assert profile.expected_task_ids == (757, 22)
    assert profile.expected_task_count == 2
    assert profile.manifest_sha256 == canonical_json_sha256(manifest)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda c: c["tasks"].append(c["tasks"][0]), "duplicate AXIS task"),
        (lambda c: c["tasks"][0].update(task_id=99999), "unknown AXIS task"),
        (lambda c: c["tasks"][0].update(task_id=True), "positive integer"),
        (lambda c: c["tasks"][0].update(instruction="wrong prompt"), "instruction differs"),
        (lambda c: c.update(tasks=[]), "non-empty list"),
        (lambda c: c.update(policy_seed=-1), "policy_seed"),
        (lambda c: c.update(protocol_revision=""), "protocol_revision"),
        (lambda c: c.update(source_manifest_sha256="0" * 64), "sha256 mismatch"),
        (lambda c: c.update(source_manifest="../manifest.json"), "JSON filename"),
        (lambda c: c.update(unknown_setting=1), "exactly these fields"),
    ],
)
def test_invalid_yaml_fails_before_evaluation(editable_config, mutation, error):
    config, path = editable_config
    mutation(config)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=error):
        load_manifest(path)


@pytest.mark.parametrize(
    "selector,expected_name",
    [
        (["--benchmark=axis_v1.0"], "axis_v1.0"),
        (["--axis_v1.0"], "axis_v1.0"),
        (["--benchmark=axis_v2.0"], "axis_v2.0"),
    ],
)
def test_run_eval_cli_uses_yaml_defaults(selector, expected_name):
    argv = ["run_eval.py", "--model", ".", "--commit-id", "local", "--dry-run", *selector]
    with mock.patch.object(sys, "argv", argv), mock.patch.object(axis_backend, "run", return_value=0) as run:
        with mock.patch.object(run_eval, "AXIS_VENV_PY"), contextlib.redirect_stdout(io.StringIO()):
            run_eval.main()
    args = run.call_args.args[0]
    assert args.benchmark == expected_name
    assert args.config == "pi05_axis_joint"
    assert args.axis_replan_steps == 10
    assert args.seed == 20260907
    assert args.workers_per_gpu == 1
    assert len(args.axis_loaded_manifest["tasks"]) == 30


def test_run_eval_rejects_conflicting_selectors_and_unsupported_backbone():
    for flags in (
        ["--axis_v1.0", "--benchmark=libero"],
        ["--axis_v1.0", "--backbone=lingbot-vla-2.0"],
    ):
        argv = ["run_eval.py", "--model", ".", "--commit-id", "local", "--dry-run", *flags]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
            with pytest.raises(SystemExit) as exc:
                run_eval.main()
        assert exc.value.code == 2


@pytest.mark.parametrize(
    "selector",
    ["--benchmark=axis-v0.1", "--benchmark=axis-v0.2", "--benchmark=axis_v0.2", "--benchmark=axis_0.2", "--axis_v0.2"],
)
def test_retired_benchmark_selectors_fail_before_evaluation(selector):
    argv = ["run_eval.py", "--model", ".", "--commit-id", "local", selector]
    with mock.patch.object(sys, "argv", argv), mock.patch.object(axis_backend, "run") as run:
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit) as exc:
            run_eval.main()
    assert exc.value.code == 2
    run.assert_not_called()
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit) as exc:
        worker_args(selector)
    assert exc.value.code == 2


@pytest.mark.parametrize("name", ["axis-v0.1", "axis-v0.2", "axis_v0.2", "axis_0.2"])
def test_retired_queue_benchmarks_do_not_get_v1_scores(name):
    profiles.configure_axis_profiles()
    with pytest.raises(profiles.BenchmarkNotReadyError, match="unknown benchmark profile"):
        worker.select_benchmark({"benchmark": name})
    assert profiles.get_profile("axis_v1.0").expected_task_count == 30


def test_children_use_a_reproducible_manifest_and_scene_snapshot(tmp_path):
    manifest = load_manifest(AXIS_V1_CONFIG_PATH)
    args = types.SimpleNamespace(
        benchmark="axis_v1.0",
        axis_manifest=None,
        task_ids="22",
        num_trials=None,
        axis_replan_steps=10,
        axis_cache_root=str(tmp_path / "cache"),
        axis_asset_fetch_workers=2,
        axis_policy_host="127.0.0.1",
        axis_policy_port=None,
        base_port=9000,
        axis_task_api_base_url=None,
        axis_asset_base_url=None,
        axis_max_control_steps=None,
        task_timeout=60,
        suites=None,
        tasks=None,
        init_seed=None,
        init_states_root=None,
        output_dir=str(tmp_path / "output"),
        dry_run=True,
        model=".",
        model_family="openpi",
        backbone="pi0.5",
        commit_id="local",
        evaluator_source_git_commit=None,
        seed=20260907,
    )
    axis_python = tmp_path / "python"
    axis_python.touch()
    result = {"status": "ok", "benchmark": "axis_v1.0", "task_id": 22}
    with mock.patch.object(axis_backend, "_run_task", return_value=result) as run:
        assert axis_backend.run(args, None, [0], axis_python) == 0
    command = run.call_args.args[0]
    snapshot = pathlib.Path(command[command.index("--manifest") + 1])
    assert snapshot == tmp_path / "output/benchmark_manifest.json"
    assert load_manifest(snapshot) == manifest
    assert (snapshot.parent / manifest["task_snapshot_root"] / "22.json").is_file()
    summary = json.loads((snapshot.parent / "summary.json").read_text())
    assert summary["benchmark"] == "axis_v1.0"
    assert summary["manifest_canonical_sha256"] == canonical_json_sha256(manifest)


def worker_args(selector="--axis_v1.0"):
    argv = [
        "worker.py",
        "--backend-url",
        "http://localhost:8001",
        "--public-api-key",
        "test-public",
        "--admin-api-key",
        "test-admin",
        "--download-strategies",
        "hfd",
        "--num-trials",
        "20",
        selector,
    ]
    with mock.patch.object(sys, "argv", argv):
        return worker.parse_args()


def complete_summary(name="axis_v1.0"):
    profile = profiles.get_profile(name)
    return dict(
        benchmark=profile.name,
        protocol_revision=profile.protocol_revision,
        manifest_canonical_sha256=profile.manifest_sha256,
        policy_seed=profile.policy_seed,
        num_trials_per_task=20,
        dry_run=False,
        tasks={
            str(task_id): dict(
                benchmark=profile.name,
                task_id=task_id,
                status="ok",
                num_trials=20,
                num_successes=10,
                success_rate=0.5,
                episodes=[{"duration_s": 1.0}] * 20,
            )
            for task_id in TASK_IDS
        },
        suites={profile.name: dict(tasks=30, episodes=600, successes=300, success_rate=0.5)},
    )


@pytest.mark.parametrize(
    "name,selectors",
    [
        ("axis_v1.0", ("--benchmark=axis_v1.0", "--axis_v1.0")),
    ],
)
def test_worker_cli_routing_and_complete_score(tmp_path, name, selectors):
    for selector in selectors:
        args = worker_args(selector)
        assert args.benchmark == name
        assert args.eval_config == "pi05_axis_joint"
        assert args.no_init_randomization
        assert args.workers_per_gpu == 1
    profile = profiles.get_profile(args.benchmark)
    task = {"base_model": "pi0.5", "benchmark": args.benchmark, "protocol_revision": profile.protocol_revision}
    assert worker.select_benchmark(task) == args.benchmark
    assert worker.select_benchmark({}, args.benchmark) == args.benchmark
    assert worker.select_benchmark({**task, "benchmark": "axis_v1.0"}, args.benchmark) == args.benchmark
    with pytest.raises(ValueError, match="protocol_revision"):
        worker.select_benchmark({**task, "protocol_revision": "old"})
    with pytest.raises(ValueError, match="require base_model"):
        worker.select_base_model({"base_model": "lingbot-vla-2.0"}, args.benchmark)
    (tmp_path / "summary.json").write_text(json.dumps(complete_summary(name)))
    proc = mock.Mock(returncode=0)
    proc.poll.return_value = 0
    worker.stop_event.clear()
    with mock.patch.object(worker.subprocess, "Popen", return_value=proc) as popen:
        summary, error = worker.run_evaluation(task, tmp_path / "model", tmp_path, args)
    assert not error
    command = popen.call_args.args[0]
    assert command[command.index("--benchmark") + 1] == name
    assert command[command.index("--backbone") + 1] == "pi0.5"
    assert command[command.index("--seed") + 1] == "20260907"
    payload = build_score_payload(task, summary, 100, benchmark=args.benchmark)
    assert payload["success"]
    assert len(payload["per_task_scores"]) == 30
    assert payload["total_score"] == 0.5
    assert payload["protocol_revision"] == profile.protocol_revision


@pytest.mark.parametrize(
    "field,value",
    [
        ("manifest_canonical_sha256", "wrong"),
        ("benchmark", "axis_v9.9"),
        ("protocol_revision", "old"),
        ("policy_seed", 1),
        ("dry_run", True),
    ],
)
def test_worker_rejects_a_different_config_or_protocol(tmp_path, field, value):
    summary = complete_summary()
    summary[field] = value
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    proc = mock.Mock(returncode=0)
    proc.poll.return_value = 0
    worker.stop_event.clear()
    with mock.patch.object(worker.subprocess, "Popen", return_value=proc):
        with pytest.raises(worker.EvalInfrastructureError, match="does not match"):
            worker.run_evaluation({"base_model": "pi0.5"}, tmp_path / "model", tmp_path, worker_args())


@pytest.mark.parametrize(
    "flags",
    [
        ["--num-trials", "1"],
        ["--task-ids", "501"],
        ["--eval-config", "pi05_libero"],
        ["--server-impl", "batched"],
    ],
)
def test_worker_rejects_development_protocol_overrides(flags):
    argv = [
        "worker.py",
        "--backend-url",
        "http://localhost:8001",
        "--api-key",
        "test",
        "--download-strategies",
        "hfd",
        "--num-trials",
        "20",
        "--axis_v1.0",
        *flags,
    ]
    with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
        with pytest.raises(SystemExit) as exc:
            worker.parse_args()
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda summary: summary["tasks"].pop("22"),
        lambda summary: summary["tasks"]["22"].update(task_id=999),
        lambda summary: summary["tasks"]["22"].update(num_trials=1),
    ],
)
def test_worker_rejects_incomplete_or_wrong_task_results(tmp_path, mutation):
    summary = complete_summary()
    mutation(summary)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    proc = mock.Mock(returncode=0)
    proc.poll.return_value = 0
    worker.stop_event.clear()
    with mock.patch.object(worker.subprocess, "Popen", return_value=proc):
        with pytest.raises(worker.EvalInfrastructureError):
            worker.run_evaluation({"base_model": "pi0.5"}, tmp_path / "model", tmp_path, worker_args())
