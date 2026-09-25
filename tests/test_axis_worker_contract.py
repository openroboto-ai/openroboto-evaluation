"""Exercise AXIS orchestration through real local HTTP progress/score requests.

Policy and simulation processes are replaced with completed trial results;
no GPU, model download, or deployed Prototype service is involved.
"""

import copy
import json
import pathlib
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "libero_eval"))

import axis_backend
from benchmark_worker import worker
from benchmark_worker.backend_client import BackendClient
from benchmark_worker.profiles import get_profile
from benchmark_worker.scoring import build_score_payload, prepare_submit_payload, successful_payload_incomplete_reason
from benchmark_worker.state import StateStore


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("X-API-Key"), payload))
            body = b'{"success": true}'
            self.send_response(self.server.response_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.response_status = 200
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        client = BackendClient(
            f"http://127.0.0.1:{server.server_port}", "public-test", "admin-test", timeout_s=2, submit_timeout_s=2
        )
        yield client, requests, server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def run_axis(tmp_path, monkeypatch, *, failed_prepare=None, dry_run=False):
    args = types.SimpleNamespace(
        benchmark="axis_v1.0",
        axis_manifest=None,
        task_ids=None,
        num_trials=20,
        axis_replan_steps=10,
        axis_cache_root=str(tmp_path / "cache"),
        axis_asset_fetch_workers=2,
        axis_policy_host="127.0.0.1",
        axis_policy_port=9000,
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
        progress_file=str(tmp_path / "progress.jsonl"),
        dry_run=dry_run,
        model="test/model",
        model_family="openpi",
        backbone="pi0.5",
        commit_id="a" * 40,
        evaluator_source_git_commit="b" * 40,
        seed=20260907,
        workers_per_gpu=1,
    )
    axis_python = tmp_path / "python"
    axis_python.touch()

    def evaluate(command, *args):
        task_id = int(command[command.index("--task-id") + 1])
        if "--prepare-only" in command:
            return {"status": "error", "error": "missing asset"} if task_id == failed_prepare else {"status": "ok"}
        return {
            "benchmark": "axis_v1.0",
            "task_id": task_id,
            "status": "ok",
            "num_trials": 0 if dry_run else 20,
            "num_successes": 0 if dry_run else 10,
            "success_rate": 0.0 if dry_run else 0.5,
            "duration_s": 60,
        }

    monkeypatch.setattr(axis_backend, "_run_task", evaluate)
    result = axis_backend.run(args, tmp_path / "checkpoint", [0, 1], axis_python)
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    return result, summary


@pytest.fixture
def completed(tmp_path, monkeypatch):
    result, summary = run_axis(tmp_path, monkeypatch)
    assert result == 0
    task = {
        "task_id": "submission-1",
        "miner_hotkey": "miner-1",
        "hf_repo_id": "test/model",
        "hf_commit": "a" * 40,
        "base_model": "pi0.5",
        "benchmark": "axis_v1.0",
        "protocol_revision": get_profile("axis_v1.0").protocol_revision,
    }
    payload = build_score_payload(task, summary, 1200, benchmark="axis_v1.0")
    assert payload["success"], payload["error"]
    return task, payload


def test_all_thirty_native_ids_survive_progress_and_score_http_submission(tmp_path, completed, backend):
    task, payload = completed
    client, requests, _ = backend
    worker._forward_progress_events(
        tmp_path / "progress.jsonl",
        0,
        lambda stage, detail: worker._report_progress(client, task["task_id"], stage, detail, "axis-worker-1"),
    )
    progress = [body for path, _, body in requests if path == "/api/benchmark-progress"]
    assert progress[0]["detail"]["tasks_done"] == 0
    assert progress[0]["detail"]["episodes_total"] == 600
    assert progress[-1]["detail"]["tasks_done"] == 30
    assert progress[-1]["detail"]["episodes_done"] == 600
    expected_ids = {str(task_id) for task_id in get_profile("axis_v1.0").expected_task_ids}
    assert {item["detail"]["last_completed_task_id"] for item in progress[1:]} == expected_ids
    assert all(item["task_id"] == "submission-1" and item["stage"] == "running" for item in progress)

    store = StateStore(tmp_path / "state.json")
    store.update(task["task_id"], status="done_pending_submit", task=task, payload=payload)
    assert worker.try_submit(client, store, task["task_id"], payload)
    path, key, body = requests[-1]
    assert path == "/api/v1/benchmark/task/submission-1/score"
    assert key == "admin-test"
    assert body["benchmark"] == "axis_v1.0"
    assert body["total_score"] == body["env_scores"][0]["score"] == 0.5
    assert body["env_scores"][0]["samples"] == 600
    assert len(body["per_task_scores"]) == 30
    assert {item["task_id"] for item in body["per_task_scores"]} == expected_ids
    assert all(item["trials"] == 20 for item in body["per_task_scores"])
    assert store.get(task["task_id"])["status"] == "submitted"
    assert all(key == "admin-test" for _, key, _ in requests)


def test_retry_retains_all_task_scores_in_state_and_http(tmp_path, completed, backend):
    task, payload = completed
    client, requests, server = backend
    store = StateStore(tmp_path / "state.json")
    store.update(task["task_id"], status="done_pending_submit", task=task, payload=payload)
    server.response_status = 401
    assert not worker.try_submit(client, store, task["task_id"], payload)
    persisted = StateStore(tmp_path / "state.json").get(task["task_id"])
    assert persisted["status"] == "done_pending_submit"
    assert persisted["payload"]["per_task_scores"] == payload["per_task_scores"]
    server.response_status = 200
    assert worker.try_submit(client, store, task["task_id"], persisted["payload"])
    assert requests[-1][2] == requests[-2][2]
    assert len(requests[-1][2]["per_task_scores"]) == 30


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "old_prefix", "unknown", "zero_padded", "integer"])
def test_invalid_or_old_cached_ids_are_requeued_before_http(tmp_path, completed, backend, invalid):
    task, original = completed
    payload = copy.deepcopy(original)
    entries = payload["per_task_scores"]
    if invalid == "missing":
        entries.pop()
    else:
        entries[0]["task_id"] = {
            "duplicate": entries[1]["task_id"],
            "old_prefix": "axis_v1.0_22",
            "unknown": "999",
            "zero_padded": "022",
            "integer": 22,
        }[invalid]
    assert successful_payload_incomplete_reason(payload)
    client, requests, _ = backend
    store = StateStore(tmp_path / "state.json")
    store.update(task["task_id"], status="done_pending_submit", task=task, payload=payload)
    assert not worker.try_submit(client, store, task["task_id"], payload)
    assert requests == []
    assert store.get(task["task_id"])["status"] == "pending"


def test_legacy_round_is_removed_without_losing_axis_details(completed):
    _, original = completed
    payload = {**original, "round_num": 42}
    submitted = prepare_submit_payload(payload)
    assert "round_num" not in submitted
    assert len(submitted["per_task_scores"]) == 30
    assert payload["round_num"] == 42


def test_preparation_failure_is_visible_and_never_counted_as_twenty_trials(tmp_path, monkeypatch):
    result, _ = run_axis(tmp_path, monkeypatch, failed_prepare=22)
    assert result == 2
    detail = json.loads((tmp_path / "progress.jsonl").read_text().splitlines()[-1])["detail"]
    assert detail["tasks_done"] == 30
    assert detail["tasks_failed"] == 1
    assert detail["episodes_done"] == 580


def test_environment_dry_run_does_not_report_model_evaluation_progress(tmp_path, monkeypatch):
    result, _ = run_axis(tmp_path, monkeypatch, dry_run=True)
    assert result == 0
    assert not (tmp_path / "progress.jsonl").exists()
