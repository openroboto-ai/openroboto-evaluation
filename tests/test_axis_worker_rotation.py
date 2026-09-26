"""Rotation contract, immutable hot loading, and queue-to-score integration."""

import copy
import fcntl
import json
import pathlib
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmark_worker import axis_rotation, profiles, worker
from benchmark_worker.axis_rotation import AxisRotation, default_rotation_directory, normalize_rotation
from benchmark_worker.backend_client import BackendClient, BackendError
from benchmark_worker.scoring import prepare_submit_payload
from benchmark_worker.state import StateStore
from libero_eval.axis_runtime import AXIS_V1_CONFIG_PATH, load_manifest
import test_axis_selector_sync
from test_worker_queue_benchmark import parse

inputs = test_axis_selector_sync.inputs

REQUEST = {
    "benchmark": "axis_v1.1",
    "previous_benchmark": "axis_v1.0",
    "seed": "0x1234",
    "seed_block": 6123456,
    "opens_at": "2026-10-15T00:00:00Z",
}


@pytest.fixture(autouse=True)
def reset_profiles():
    profiles.configure_axis_profiles()
    yield
    profiles.configure_axis_profiles()


@pytest.fixture
def rotation(inputs, tmp_path):
    directory = tmp_path / "benchmarks"
    profiles.configure_axis_profiles(directory)
    return AxisRotation(
        directory=directory,
        backend_url="http://test-backend",
        selector_root=inputs["selector_root"],
        runtime_pool=inputs["runtime_pool"],
    )


def summary_for(name):
    profile = profiles.get_profile(name)
    count = profile.expected_task_count
    return {
        "benchmark": name,
        "protocol_revision": profile.protocol_revision,
        "manifest_canonical_sha256": profile.manifest_sha256,
        "policy_seed": profile.policy_seed,
        "num_trials_per_task": 20,
        "dry_run": False,
        "tasks": {
            str(tid): {
                "benchmark": name,
                "task_id": tid,
                "status": "ok",
                "num_trials": 20,
                "num_successes": 10,
                "success_rate": 0.5,
                "episodes": [{"duration_s": 1.0}] * 20,
            }
            for tid in profile.expected_task_ids
        },
        "suites": {name: {"tasks": count, "episodes": count * 20, "successes": count * 10, "success_rate": 0.5}},
    }


def test_unknown_version_is_discovered_without_restart_and_old_version_remains(rotation):
    with pytest.raises(ValueError, match="unknown benchmark"):
        profiles.get_profile("axis_v1.1")
    path = rotation.prepare(REQUEST)
    profile = profiles.get_profile("axis_v1.1")
    assert profile.name == "axis_v1.1" and profile.runtime_benchmark == "axis"
    assert profile.manifest_path == path
    assert profile.expected_task_count == 40
    assert profiles.get_profile("axis_v1.0").expected_task_count == 30
    assert worker._protocol_revision("axis_v1.1") == profile.protocol_revision
    second = rotation.prepare({
        **REQUEST,
        "benchmark": "axis_v1.2",
        "previous_benchmark": "axis_v1.1",
        "seed": "0x5678",
    })
    assert len(load_manifest(second)["tasks"]) == 50
    assert profiles.get_profile("axis_v1.1") == profile


def test_manual_bundle_arriving_after_unknown_queue_task_is_loaded(inputs, tmp_path):
    profiles.configure_axis_profiles(tmp_path)
    with pytest.raises(ValueError):
        worker.select_benchmark({"benchmark": "axis_v1.1"})
    axis_rotation.sync(**{**inputs, "previous": AXIS_V1_CONFIG_PATH}, output=tmp_path / "axis_v1.1")
    assert worker.select_benchmark({"benchmark": "axis_v1.1"}) == "axis_v1.1"


def test_idempotent_preparation_rejects_changed_seed_and_keeps_original_bundle(rotation):
    path = rotation.prepare(REQUEST)
    files = {p: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
    assert rotation.prepare({**REQUEST, "opens_at": "2026-10-16T00:00:00Z"}) == path
    assert all(p.read_bytes() == raw for p, raw in files.items())
    with pytest.raises(ValueError, match="requested seed"):
        rotation.prepare({**REQUEST, "seed": "0x1235"})
    with pytest.raises(ValueError, match="not ready"):
        profiles.get_profile("axis_v1.1")
    assert all(p.read_bytes() == raw for p, raw in files.items())
    assert rotation.prepare(REQUEST) == path


@pytest.mark.parametrize("change", ["payload", "yaml", "missing"])
def test_registered_version_cannot_change_or_disappear(rotation, change):
    path = rotation.prepare(REQUEST)
    if change == "payload":
        (path.parent / "axis_v1.1-tasks/22.json").write_text("{}")
    elif change == "yaml":
        path.write_text(path.read_text().replace("policy_seed: 20260907", "policy_seed: 20260908"))
    else:
        path.unlink()
    profiles.refresh_axis_profiles()
    with pytest.raises(ValueError, match="not ready"):
        profiles.get_profile("axis_v1.1")
    assert profiles.get_profile("axis_v1.0").expected_task_count == 30


def test_backend_isolation_and_pins_persist_across_restart(rotation):
    assert default_rotation_directory("https://api-dev.example") != default_rotation_directory("https://api.example")
    args = dict(directory=rotation.directory, selector_root=rotation.selector_root, runtime_pool=rotation.runtime_pool)
    with pytest.raises(ValueError, match="different backend"):
        AxisRotation(**args, backend_url="http://another-backend")
    rotation.prepare(REQUEST)
    restarted = AxisRotation(**args, backend_url="http://test-backend/")
    assert restarted.prepare(REQUEST).exists()
    code = rotation.selector_root / "selector.py"
    code.write_text(code.read_text() + "\n# changed selector\n")
    with pytest.raises(ValueError, match="changed while"):
        rotation.prepare(REQUEST)
    with pytest.raises(ValueError, match="pinned selector"):
        AxisRotation(**args, backend_url="http://test-backend")


def test_archived_bundle_is_verified_against_its_receipt_after_restart(rotation):
    path = rotation.prepare(REQUEST)
    path.write_text(path.read_text() + "\n# edited after publication\n")
    profiles.configure_axis_profiles(rotation.directory)
    with pytest.raises(ValueError, match="frozen selection receipt"):
        profiles.get_profile("axis_v1.1")


def test_generation_failure_and_concurrent_preparer_do_not_publish_partial_version(rotation, monkeypatch):
    with (rotation.directory / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert rotation.prepare(REQUEST) is None
    real_sync = axis_rotation.sync

    def fail(**kwargs):
        kwargs["output"].mkdir()
        (kwargs["output"] / "axis_v1.1.yaml").write_text("incomplete")
        raise OSError("simulated interrupted generation")

    monkeypatch.setattr(axis_rotation, "sync", fail)
    with pytest.raises(OSError, match="interrupted"):
        rotation.prepare(REQUEST)
    assert not (rotation.directory / "axis_v1.1").exists()
    assert not list(rotation.directory.glob(".preparing-*"))
    with pytest.raises(ValueError, match="not ready"):
        profiles.get_profile("axis_v1.1")
    monkeypatch.setattr(axis_rotation, "sync", real_sync)
    assert rotation.prepare(REQUEST).exists()


@pytest.mark.parametrize(
    "change",
    [
        {"benchmark": "../../axis_v1.1"},
        {"benchmark": "axis_v1.2"},
        {"benchmark": "axis_v2.0"},
        {"previous_benchmark": "axis_v1.1"},
        {"seed": "not-a-hash"},
        {"seed": -1},
        {"seed_block": True},
        {"seed_block": -1},
        {"opens_at": "2026-10-15"},
        {"opens_at": None},
    ],
)
def test_invalid_rotation_never_creates_a_version(rotation, change):
    with pytest.raises(ValueError):
        normalize_rotation({**REQUEST, **change}, rotation.selector)
    assert not list(rotation.directory.glob("axis_v*"))


def test_rotation_endpoint_uses_worker_key_and_rejects_malformed_envelopes():
    client = BackendClient("http://example", "public", "admin", worker_api_key="worker")
    with mock.patch.object(client, "_request", return_value={"data": REQUEST}) as request:
        assert client.fetch_rotation() == REQUEST
    request.assert_called_once_with("GET", "/api/v1/benchmark/rotation", api_key="worker")
    with mock.patch.object(client, "_request", return_value={"data": None}):
        assert client.fetch_rotation() is None
    for response in ({}, {"data": []}, []):
        with mock.patch.object(client, "_request", return_value=response), pytest.raises(BackendError):
            client.fetch_rotation()
    with pytest.raises(BackendError, match="WORKER_KEY"):
        BackendClient("http://example", "public", "admin").fetch_rotation()


def test_queued_version_becoming_unavailable_waits_for_poll_without_writes(rotation, tmp_path, monkeypatch):
    rotation.prepare(REQUEST)
    args = parse(monkeypatch, "--axis-benchmark-dir", str(rotation.directory))
    task = {"task_id": "waiting", "benchmark": "axis_v1.1"}
    store = StateStore(tmp_path / "state.json")
    store.update("waiting", status="pending", task=task)
    local_queue = queue.Queue()
    local_queue.put(task)
    local_queue.put(None)
    profiles.block_axis_profile("axis_v1.1", "configuration needs correction")
    client = mock.Mock()
    monkeypatch.setattr(worker, "wait_for_gpu_health", lambda: True)
    worker.stop_event.clear()
    worker.worker_loop(local_queue, client, store, args)
    assert local_queue.empty()
    assert store.get("waiting")["status"] == "stale"
    client.report_progress.assert_not_called()
    client.submit_score.assert_not_called()


@pytest.mark.parametrize(
    "extra",
    [
        ["--axis-selector-root", "/selector"],
        ["--axis-runtime-pool", "/runtime.json"],
        ["--axis-selector-root", "/selector", "--axis-runtime-pool", "/runtime.json"],
        [
            "--axis-selector-root",
            "/selector",
            "--axis-runtime-pool",
            "/runtime.json",
            "--worker-key",
            "key",
            "--benchmark",
            "axis_v1.0",
        ],
    ],
)
def test_rotation_requires_complete_configuration_and_queue_routing(monkeypatch, extra):
    monkeypatch.delenv("WORKER_KEY", raising=False)
    with pytest.raises(SystemExit):
        parse(monkeypatch, *extra)


def test_http_rotation_waits_then_evaluates_and_reports_new_version_without_restart(inputs, tmp_path, monkeypatch):
    requests, commands, scores = [], [], []
    polls = 0
    task = {
        "task_id": "baseline-v11",
        "benchmark": "axis_v1.1",
        "hf_repo_id": "test/model",
        "hf_commit": "a" * 40,
        "base_model": "pi0.5",
        "miner_hotkey": "baseline",
    }

    class Handler(BaseHTTPRequestHandler):
        def respond(self, value):
            raw = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            nonlocal polls
            requests.append((self.path, self.headers.get("X-API-Key"), None))
            if self.path.endswith("/rotation"):
                polls += 1
                if polls > 10:
                    worker.stop_event.set()
                self.respond({"data": None if polls == 1 else REQUEST})
            else:
                self.respond({"tasks": [task]})

        def do_POST(self):
            value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("X-API-Key"), value))
            assert polls >= 2  # No progress/score writes while the version is unknown.
            if self.path.endswith("/score"):
                scores.append(value)
                worker.stop_event.set()
            self.respond({"success": True})

        def log_message(self, *_):
            pass

    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()

    def launch(command, **kwargs):
        commands.append(command)
        assert command[command.index("--benchmark") + 1] == "axis"
        manifest = load_manifest(pathlib.Path(command[command.index("--axis-manifest") + 1]))
        assert manifest["name"] == "axis_v1.1" and len(manifest["tasks"]) == 40
        output = pathlib.Path(command[command.index("--output-dir") + 1])
        (output / "summary.json").write_text(json.dumps(summary_for("axis_v1.1")))
        return mock.Mock(returncode=0, poll=lambda: 0)

    try:
        args = parse(
            monkeypatch,
            "--backend-url",
            f"http://127.0.0.1:{server.server_port}",
            "--axis-selector-root",
            str(inputs["selector_root"]),
            "--axis-runtime-pool",
            str(inputs["runtime_pool"]),
            "--axis-benchmark-dir",
            str(tmp_path / "rounds"),
            "--worker-key",
            "worker-test",
            "--poll-interval",
            "0.01",
        )
        args.state_file = str(tmp_path / "state.json")
        args.output_root, args.download_dir = tmp_path / "runs", tmp_path / "models"
        monkeypatch.setattr(worker, "parse_args", lambda: copy.copy(args))
        monkeypatch.setattr(worker, "resolve_evaluator_source_commit", lambda _: "b" * 40)
        monkeypatch.setattr(worker, "preflight_runtime", lambda _: None)
        monkeypatch.setattr(worker, "wait_for_gpu_health", lambda: True)
        monkeypatch.setattr(worker, "_setup_logger", lambda _: tmp_path / "worker.log")
        monkeypatch.setattr(worker, "_install_stderr_logging", lambda: None)
        monkeypatch.setattr(worker, "download_with_retry", lambda *_, **__: None)
        monkeypatch.setattr(worker.subprocess, "Popen", launch)
        worker.stop_event.clear()
        worker.main()
        assert polls >= 2 and len(commands) == len(scores) == 1
        score = scores[0]
        assert score["success"] and score["benchmark"] == "axis_v1.1"
        assert score["env_scores"][0]["env_name"] == "axis_v1.1"
        assert {r["task_id"] for r in score["per_task_scores"]} == {
            str(tid) for tid in profiles.get_profile("axis_v1.1").expected_task_ids
        }
        assert prepare_submit_payload(score)["per_task_scores"] == score["per_task_scores"]
        assert StateStore(pathlib.Path(args.state_file)).get(task["task_id"])["status"] == "submitted"
        assert all(key == "worker-test" for path, key, _ in requests if path.endswith("/rotation"))
        assert all(key == "admin-test" for _, key, value in requests if value is not None)
    finally:
        worker.stop_event.clear()
        server.shutdown()
        thread.join()
        server.server_close()
