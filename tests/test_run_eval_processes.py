"""GPU admission and bounded policy-server cleanup regressions (no GPU needed)."""

import pathlib
import subprocess
import sys
import types
from contextlib import ExitStack
from unittest import mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "libero_eval"))
import run_eval
from gpu_health import GpuHealth


def test_busy_gpu_prevents_model_and_simulator_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--model", "unused", "--commit-id", "a" * 40, "--gpus", "4,7"])
    monkeypatch.setattr(run_eval, "CLIENT_VENV_PY", tmp_path)
    monkeypatch.setattr(run_eval, "check_gpu_health", lambda: GpuHealth(True, "ok"))
    monkeypatch.setattr(run_eval, "acquire_gpu_locks", lambda gpus: [])
    with ExitStack() as stack:
        check = stack.enter_context(
            mock.patch.object(run_eval, "check_gpu_availability", return_value=GpuHealth(False, "already in use"))
        )
        resolve = stack.enter_context(mock.patch.object(run_eval, "resolve_model"))
        start = stack.enter_context(mock.patch.object(run_eval, "start_servers"))
        with pytest.raises(SystemExit, match="GPU reservation failed: already in use"):
            run_eval.main()
    check.assert_called_once_with([4, 7])
    resolve.assert_not_called()
    start.assert_not_called()


def test_killed_server_is_reaped():
    proc = mock.Mock(pid=123)
    proc.poll.return_value = None
    proc.wait.side_effect = [subprocess.TimeoutExpired("server", 10), -9]
    run_eval.stop_servers([types.SimpleNamespace(proc=proc, gpu=4)])
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert proc.wait.call_count == 2
    assert 0 <= proc.wait.call_args_list[-1].kwargs["timeout"] <= 5


def test_all_unkillable_servers_share_cleanup_deadlines(monkeypatch, capsys):
    clock = [0.0]
    monkeypatch.setattr(run_eval.time, "monotonic", lambda: clock[0])
    servers = []
    for pid in range(7):
        proc = mock.Mock(pid=pid)
        proc.poll.return_value = None

        def wait(*, timeout):
            clock[0] += timeout
            raise subprocess.TimeoutExpired("server", timeout)

        proc.wait.side_effect = wait
        servers.append(types.SimpleNamespace(proc=proc, gpu=pid))
    run_eval.stop_servers(servers)
    assert clock[0] == 15
    assert capsys.readouterr().err.count("did not exit after SIGKILL") == 7


def test_server_exit_race_does_not_skip_other_children():
    first = mock.Mock(pid=123)
    first.poll.return_value = None
    first.terminate.side_effect = ProcessLookupError
    second = mock.Mock(pid=124)
    second.poll.return_value = None
    run_eval.stop_servers([types.SimpleNamespace(proc=first, gpu=4), types.SimpleNamespace(proc=second, gpu=5)])
    second.terminate.assert_called_once()
    first.wait.assert_called()
    second.wait.assert_called()


def test_real_server_child_is_reaped():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        run_eval.stop_servers([types.SimpleNamespace(proc=proc, gpu=4)])
        assert proc.returncode is not None
        assert not pathlib.Path(f"/proc/{proc.pid}").exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
