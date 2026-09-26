"""Queue routing through CLI resolution, subprocess dispatch, and HTTP scoring."""

import contextlib
import copy
import io
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmark_worker import worker
from benchmark_worker.profiles import get_profile
from benchmark_worker.state import StateStore
from test_axis_yaml_benchmark import complete_summary


def parse(monkeypatch, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker",
            "--backend-url",
            "http://localhost:8001",
            "--public-api-key",
            "public-test",
            "--admin-api-key",
            "admin-test",
            "--download-strategies",
            "hfd-mirror,hub-mirror",
            "--num-trials",
            "20",
            *extra,
        ],
    )
    return worker.parse_args()


@pytest.mark.parametrize("value", [None, "", "unknown", 42, [], {}])
def test_queue_requires_supported_benchmark(value):
    with pytest.raises(ValueError):
        worker.select_benchmark({"benchmark": value})


def test_explicit_override_ignores_queue_routing_metadata():
    for task in ({}, {"benchmark": "unknown"}, {"benchmark": "axis_v1.0", "protocol_revision": "old"}):
        assert worker.select_benchmark(task, "libero_pro_custom_1") == "libero_pro_custom_1"


def test_per_task_defaults_are_isolated(monkeypatch):
    args = parse(monkeypatch)
    assert args.benchmark is None
    assert args.workers_per_gpu is None
    axis = worker.task_evaluation_args({"benchmark": "axis_v1.0"}, args)
    libero = worker.task_evaluation_args({"benchmark": "libero_pro_custom_1"}, args)
    assert axis.workers_per_gpu == 1
    assert axis.eval_config == "pi05_axis_joint"
    assert axis.no_init_randomization
    assert libero.workers_per_gpu == 3
    assert libero.eval_config is None
    assert not libero.no_init_randomization
    assert args.benchmark is None and args.workers_per_gpu is None
    explicit = parse(monkeypatch, "--workers-per-gpu", "2")
    assert worker.task_evaluation_args({"benchmark": "axis_v1.0"}, explicit).workers_per_gpu == 2


def test_axis_only_preserves_queue_version_and_rejects_overrides(monkeypatch):
    args = parse(monkeypatch, "--axis-only")
    for name in ("axis_v1.0",):
        resolved = worker.task_evaluation_args({"benchmark": name}, args)
        assert resolved.benchmark == name
        assert resolved.num_trials == 20
        assert resolved.eval_config == "pi05_axis_joint"
    for name in (None, "", 42, {}, "libero_pro_custom_1", "robotwin"):
        with pytest.raises(ValueError, match="--axis-only skips"):
            worker.task_evaluation_args({"benchmark": name}, args)
    assert worker.task_matches_filter({"benchmark": "axis_v1.1"}, args)
    with pytest.raises(ValueError):
        worker.task_evaluation_args({"benchmark": "axis_v1.1"}, args)
    for override in (["--benchmark", "axis_v1.0"], ["--axis_v1.0"]):
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            parse(monkeypatch, "--axis-only", *override)


@pytest.mark.parametrize(
    "extra",
    [
        ["--num-trials", "5"],
        ["--task-ids", "22"],
        ["--server-impl", "batched"],
        ["--eval-config", "pi05_libero"],
    ],
)
def test_queue_axis_validates_before_download(monkeypatch, extra):
    args = parse(monkeypatch, *extra)
    with pytest.raises(ValueError):
        worker.task_evaluation_args({"benchmark": "axis_v1.0"}, args)
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        parse(monkeypatch, *extra, "--benchmark", "axis_v1.0")


def test_pending_scores_use_each_stored_task_benchmark():
    for name in ("libero_pro_custom_1", "axis_v1.0"):
        entry = {
            "task": {"benchmark": name},
            "benchmark": name,
            "protocol_revision": worker._protocol_revision(name),
        }
        assert worker.pending_score_matches_profile(entry, None)
        assert not worker.pending_score_matches_profile(entry, "libero")
    assert not worker.pending_score_matches_profile({**entry, "protocol_revision": "old"}, None)
    assert not worker.pending_score_matches_profile({**entry, "task": {}}, None)


@pytest.mark.parametrize("override,axis_only", [(False, False), (True, False), (False, True)])
def test_http_queue_dispatch_score_and_restart(tmp_path, monkeypatch, override, axis_only):
    task = {
        "task_id": "queue-axis",
        "benchmark": "libero" if override else "axis_v1.0",
        "hf_repo_id": "test/model",
        "hf_commit": "a" * 40,
        "base_model": "pi0.5",
        "miner_hotkey": "miner-test",
    }
    if override:
        task["protocol_revision"] = "ignored-queue-revision"
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def respond(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            requests.append((self.path, self.headers.get("X-API-Key"), None))
            tasks = [task]
            if axis_only:
                tasks.append({**task, "task_id": "old-libero", "benchmark": "libero_pro_custom_1"})
            self.respond({"tasks": tasks})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("X-API-Key"), body))
            self.respond({"success": True})

        def log_message(self, *_):
            pass

    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        out_dir = pathlib.Path(command[command.index("--output-dir") + 1])
        (out_dir / "summary.json").write_text(json.dumps(complete_summary()))
        return mock.Mock(returncode=0, poll=lambda: 0)

    try:
        args = parse(
            monkeypatch,
            "--once",
            *(["--benchmark", "axis_v1.0"] if override else []),
            *(["--axis-only"] if axis_only else []),
        )
        args.backend_url = f"http://127.0.0.1:{server.server_port}"
        args.state_file = str(tmp_path / "state.json")
        args.output_root = tmp_path / "runs"
        args.download_dir = tmp_path / "models"
        store = StateStore(pathlib.Path(args.state_file))
        retired = {}
        if axis_only:
            for status in ("pending", "running", "done_pending_submit"):
                tid = f"retired-{status}"
                store.update(
                    tid,
                    status=status,
                    task={**task, "task_id": tid, "benchmark": "libero_pro_custom_1"},
                    benchmark="libero_pro_custom_1",
                    payload={"benchmark": "libero_pro_custom_1", "success": True},
                )
                retired[tid] = store.get(tid)
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
        entry = StateStore(pathlib.Path(args.state_file)).get(task["task_id"])
        assert entry["benchmark"] == "axis_v1.0"
        assert entry["status"] == "submitted"
        assert entry["task"] == task
        assert len(commands) == 1
        cmd = commands[0]
        for flag, value in (
            ("--benchmark", "axis_v1.0"),
            ("--config", "pi05_axis_joint"),
            ("--num-trials", "20"),
            ("--workers-per-gpu", "1"),
            ("--seed", "20260907"),
        ):
            assert cmd[cmd.index(flag) + 1] == value
        scores = [(key, body) for path, key, body in requests if path.endswith("/score")]
        assert len(scores) == 1
        key, body = scores[0]
        assert key == "admin-test"
        assert body["benchmark"] == "axis_v1.0"
        assert len(body["per_task_scores"]) == 30
        assert body["protocol_revision"] == get_profile("axis_v1.0").protocol_revision
        assert all(key == "public-test" for _, key, body in requests if body is None)
        worker.main()
        assert len(commands) == 1
        assert len([path for path, _, _ in requests if path.endswith("/score")]) == 1
        if axis_only:
            final_store = StateStore(pathlib.Path(args.state_file))
            assert final_store.get("old-libero") is None
            assert all(final_store.get(tid) == entry for tid, entry in retired.items())
            assert all(
                body.get("task_id") == task["task_id"]
                for path, _, body in requests
                if path == "/api/benchmark-progress"
            )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        worker.stop_event.clear()
