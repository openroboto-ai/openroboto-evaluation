"""Task draws and their evaluator handoff; no model, simulator or GPU required."""

import contextlib
import copy
import io
import json
import pathlib
import sys
from unittest import mock

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))

import axis_backend
import run_eval
from axis_runtime import AXIS_V1_CONFIG_PATH, load_manifest
from axis_sampling import sample_tasks


@pytest.fixture
def pool():
    return load_manifest(AXIS_V1_CONFIG_PATH)


def test_draw_is_reproducible_and_without_replacement(pool):
    first = sample_tasks(pool, 8, 20260909)
    assert first == sample_tasks(pool, 8, 20260909)
    assert len(set(first["selected_task_ids"])) == 8
    assert set(first["selected_task_ids"]) <= set(first["pool_task_ids"])
    assert first["selected_task_ids"] != sample_tasks(pool, 8, 20260910)["selected_task_ids"]
    assert set(sample_tasks(pool, 30, 1)["selected_task_ids"]) == set(first["pool_task_ids"])


def test_input_order_does_not_choose_the_tasks(pool):
    shuffled = copy.deepcopy(pool)
    shuffled["tasks"].reverse()
    assert sample_tasks(pool, 8, 19)["selected_task_ids"] == sample_tasks(shuffled, 8, 19)["selected_task_ids"]
    assert sample_tasks(pool, 8, 19)["pool_manifest_sha256"] != sample_tasks(shuffled, 8, 19)["pool_manifest_sha256"]


def test_automatic_seed_is_recorded_for_replay(pool):
    draw = sample_tasks(pool, 8)
    assert draw == sample_tasks(pool, 8, draw["sampling_seed"])


@pytest.mark.parametrize("count", [0, -1, 31, True, 2.5])
def test_invalid_size_fails(pool, count):
    with pytest.raises(ValueError, match="sample count"):
        sample_tasks(pool, count, 0)


@pytest.mark.parametrize("seed", [-1, 2**64, True, 0.5, "1"])
def test_invalid_seed_fails(pool, seed):
    with pytest.raises(ValueError, match="sampling seed"):
        sample_tasks(pool, 8, seed)


@pytest.mark.parametrize(
    "flags",
    [
        ["--benchmark", "libero", "--axis-sample-size", "8"],
        ["--axis_v1.0", "--axis-sample-size", "31"],
        ["--axis_v1.0", "--axis-sampling-seed", "1"],
        ["--axis_v1.0", "--axis-sample-size", "8", "--task-ids", "22"],
    ],
)
def test_bad_cli_fails_before_runtime_start(flags):
    argv = ["run_eval.py", "--model", ".", "--commit-id", "local", *flags]
    with mock.patch.object(sys, "argv", argv), mock.patch.object(run_eval.subprocess, "run") as launch:
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit) as exc:
            run_eval.main()
    assert exc.value.code == 2
    launch.assert_not_called()


def test_evaluator_runs_only_selected_ids_and_archives_selection(tmp_path, pool):
    visited = []

    def fake_task(command, *_args):
        task_id = int(command[command.index("--task-id") + 1])
        visited.append(task_id)
        return {"task_id": task_id, "status": "ok", "num_trials": 0, "num_successes": 0}

    argv = [
        "run_eval.py",
        "--model",
        ".",
        "--commit-id",
        "local",
        "--axis_v1.0",
        "--dry-run",
        "--axis-sample-size",
        "8",
        "--axis-sampling-seed",
        "20260909",
        "--output-dir",
        str(tmp_path),
    ]
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(sys, "argv", argv))
        stack.enter_context(mock.patch.object(run_eval, "AXIS_VENV_PY", pathlib.Path(sys.executable)))
        stack.enter_context(mock.patch.object(axis_backend, "_run_task", side_effect=fake_task))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        run_eval.main()
    selection = json.loads((tmp_path / "task_selection.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert selection == sample_tasks(pool, 8, 20260909)
    assert visited == selection["selected_task_ids"]
    assert summary["task_selection"] == selection
    assert summary["evaluation_scope"] == "sampled_tasks"
    assert set(summary["tasks"]) == {str(task_id) for task_id in visited}
