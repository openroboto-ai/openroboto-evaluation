"""AXIS progress heartbeat, completed-task accounting, and worker forwarding."""

import json
import pathlib
import sys
import threading
from unittest import mock

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_progress import report_axis_progress
from benchmark_worker.backend_client import BackendClient
from benchmark_worker.worker import _forward_progress_events


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_periodic_heartbeat_during_a_long_task_and_clean_shutdown(tmp_path):
    path = tmp_path / "progress.jsonl"
    heartbeat = threading.Event()
    dumps = json.dumps

    def observe(value):
        if threading.current_thread().name == "axis-progress":
            heartbeat.set()
        return dumps(value)

    with mock.patch("axis_progress.json.dumps", side_effect=observe):
        with report_axis_progress(path, "axis_v1.0", [22], 20, interval_s=0.01):
            assert heartbeat.wait(2), "no progress heartbeat while evaluation was running"
            reporter = next(t for t in threading.enumerate() if t.name == "axis-progress")
    assert not reporter.is_alive()
    assert len(events(path)) >= 3  # initial, heartbeat, final
    assert all(event["detail"]["tasks_done"] == 0 for event in events(path))


def test_task_counts_native_ids_and_forwarding_to_backend(tmp_path):
    path = tmp_path / "progress.jsonl"
    with report_axis_progress(path, "axis_v1.0", [22, 43, 501], 20) as record:
        # A model achieving zero successes still completed its evaluation.
        record(43, {"status": "ok", "num_trials": 20, "num_successes": 0})
        record(43, {"status": "ok", "num_trials": 20, "num_successes": 0})
        record(22, {"status": "error", "error": "preparation failed"})
        record(501, {"status": "ok", "num_trials": 20})
    details = [event["detail"] for event in events(path)]
    assert details[0] == {
        "benchmark": "axis_v1.0",
        "tasks_done": 0,
        "tasks_total": 3,
        "tasks_failed": 0,
        "episodes_done": 0,
        "episodes_total": 60,
    }
    assert details[1]["tasks_failed"] == 0
    assert details[2]["tasks_done"] == 1  # duplicate notification does not double count
    assert details[-1] == {
        "benchmark": "axis_v1.0",
        "tasks_done": 3,
        "tasks_total": 3,
        "tasks_failed": 1,
        "episodes_done": 40,
        "episodes_total": 60,
        "last_completed_task_id": "501",
    }
    client = BackendClient("http://backend.test", "public", "admin")
    with mock.patch.object(client, "_request", return_value={"success": True}) as post:
        offset = _forward_progress_events(
            path, 0, lambda stage, detail: client.report_progress("submission-1", stage, detail, "worker-1")
        )
        assert offset == path.stat().st_size
        assert post.call_count == len(details)
        post.assert_called_with(
            "POST",
            "/api/benchmark-progress",
            api_key="admin",
            body={"task_id": "submission-1", "stage": "running", "detail": details[-1], "worker_id": "worker-1"},
        )


def test_progress_stops_on_exception_without_claiming_completion(tmp_path):
    path = tmp_path / "progress.jsonl"
    with pytest.raises(RuntimeError, match="server failed"):
        with report_axis_progress(path, "axis_v1.0", [22, 31], 20) as record:
            record(22, {"status": "ok", "num_trials": 20})
            reporter = next(t for t in threading.enumerate() if t.name == "axis-progress")
            raise RuntimeError("server failed")
    assert not reporter.is_alive()
    assert events(path)[-1]["detail"]["tasks_done"] == 1
    assert events(path)[-1]["detail"]["episodes_done"] == 20


def test_disabled_progress_has_no_thread_or_io():
    with mock.patch("axis_progress.threading.Thread") as thread:
        with report_axis_progress(None, "axis_v1.0", [22], 20) as record:
            record(22, {"status": "ok", "num_trials": 20})
        thread.assert_not_called()


def test_progress_write_error_is_logged_and_does_not_discard_results(tmp_path, caplog):
    with report_axis_progress(tmp_path / "progress.jsonl", "axis_v1.0", [22], 20) as record:
        with mock.patch.object(pathlib.Path, "open", side_effect=OSError("disk full")):
            record(22, {"status": "ok", "num_trials": 20})
    assert "Could not write AXIS progress" in caplog.text
    assert "disk full" in caplog.text


def test_unknown_task_does_not_change_progress(tmp_path):
    path = tmp_path / "progress.jsonl"
    with report_axis_progress(path, "axis_v1.0", [22], 20) as record:
        with pytest.raises(ValueError, match="unselected task"):
            record(999, {"status": "ok", "num_trials": 20})
    assert events(path)[-1]["detail"]["tasks_done"] == 0
