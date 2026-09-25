import pathlib
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from unittest import mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from libero_eval import gpu_health


def test_existing_blocked_probe_prevents_more_children(tmp_path, monkeypatch):
    for pid, name, state in [(123, "nvidia-smi", "D (disk sleep)"), (124, "python", "D (disk sleep)")]:
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "status").write_text(f"Name:\t{name}\nState:\t{state}\n")
    assert gpu_health.blocked_gpu_probes(tmp_path) == [123]
    monkeypatch.setattr(gpu_health, "blocked_gpu_probes", lambda: [123])
    with mock.patch.object(gpu_health.subprocess, "Popen") as popen:
        health = gpu_health.check_gpu_health()
    assert not health.healthy
    assert "123" in health.detail
    popen.assert_not_called()


def test_unkillable_child_has_no_unbounded_wait(monkeypatch):
    monkeypatch.setattr(gpu_health, "blocked_gpu_probes", lambda: [])
    proc = mock.Mock(pid=123)
    proc.wait.side_effect = subprocess.TimeoutExpired("nvidia-smi", 15)
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(gpu_health.subprocess, "Popen", return_value=proc))
        killpg = stack.enter_context(mock.patch.object(gpu_health.os, "killpg"))
        health = gpu_health.check_gpu_health()
    assert not health.healthy
    assert "timed out" in health.detail
    assert proc.wait.call_args_list == [mock.call(timeout=15), mock.call(timeout=1)]
    killpg.assert_called_once_with(123, signal.SIGKILL)


@pytest.mark.parametrize(
    ("returncode", "output", "healthy"),
    [
        (0, "GPU 0: Test (UUID: GPU-123)\n", True),
        (0, "No devices were found\n", False),
        (1, "Driver unavailable", False),
        (0, "GPU 0: Test\nUnable to determine device handle: Unknown Error", False),
    ],
)
def test_probe_output(returncode, output, healthy, monkeypatch):
    monkeypatch.setattr(gpu_health, "blocked_gpu_probes", lambda: [])

    def launch(*args, **kwargs):
        kwargs["stdout"].write(output.encode())
        kwargs["stdout"].flush()
        return mock.Mock(wait=mock.Mock(return_value=returncode))

    with mock.patch.object(gpu_health.subprocess, "Popen", side_effect=launch):
        assert gpu_health.check_gpu_health().healthy is healthy


def test_actual_slow_child_is_reaped_within_deadline(monkeypatch):
    monkeypatch.setattr(gpu_health, "blocked_gpu_probes", lambda: [])
    real_popen = subprocess.Popen
    children = []

    def launch(*args, **kwargs):
        child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    with mock.patch.object(gpu_health.subprocess, "Popen", side_effect=launch):
        started = time.monotonic()
        health = gpu_health.check_gpu_health(timeout=0.05)
    assert not health.healthy
    assert time.monotonic() - started < 2
    assert children[0].poll() is not None


@pytest.mark.parametrize("process_type", ["C", "G", "C+G"])
def test_availability_rejects_compute_and_graphics_clients(process_type, monkeypatch):
    xml = (
        '<nvidia_smi_log><gpu id="0000:99:00.0"><processes><process_info>'
        f"<pid>456</pid><type>{process_type}</type>"
        "</process_info></processes></gpu></nvidia_smi_log>"
    )
    monkeypatch.setattr(gpu_health, "blocked_gpu_probes", lambda: [])

    def launch(command, **kwargs):
        assert command == ["nvidia-smi", "-i", "4", "-q", "-x"]
        kwargs["stdout"].write(xml.encode())
        kwargs["stdout"].flush()
        return mock.Mock(wait=mock.Mock(return_value=0))

    with mock.patch.object(gpu_health.subprocess, "Popen", side_effect=launch):
        health = gpu_health.check_gpu_availability([4])
    assert not health.healthy
    assert "pid=456" in health.detail
    assert "0000:99:00.0" in health.detail


@pytest.mark.parametrize(
    ("xml", "healthy"),
    [
        ('<nvidia_smi_log><gpu id="0000:99:00.0"><processes/></gpu></nvidia_smi_log>', True),
        ("<nvidia_smi_log/>", False),
        ("<nvidia_smi_log><gpu/></nvidia_smi_log>", False),
        ("<nvidia_smi_log><gpu><processes>N/A</processes></gpu></nvidia_smi_log>", False),
        ("<nvidia_smi_log><gpu><processes><process_info/></processes></gpu></nvidia_smi_log>", False),
        ("not XML", False),
    ],
)
def test_availability_fails_closed_on_missing_accounting(xml, healthy, monkeypatch):
    monkeypatch.setattr(gpu_health, "_probe_gpu", lambda *a, **kw: gpu_health.GpuHealth(True, xml))
    assert gpu_health.check_gpu_availability([4]).healthy is healthy


@pytest.mark.parametrize("gpus", [[], [4, 4], [-1]])
def test_availability_validates_device_selection(gpus):
    with mock.patch.object(gpu_health.subprocess, "Popen") as popen:
        with pytest.raises(ValueError):
            gpu_health.check_gpu_availability(gpus)
    popen.assert_not_called()


def test_availability_does_not_probe_an_already_blocked_driver(monkeypatch):
    monkeypatch.setattr(gpu_health, "blocked_gpu_probes", lambda: [123])
    with mock.patch.object(gpu_health.subprocess, "Popen") as popen:
        health = gpu_health.check_gpu_availability([4])
    assert not health.healthy
    popen.assert_not_called()


def test_availability_reads_large_xml_and_rejects_truncation(monkeypatch):
    monkeypatch.setattr(gpu_health, "blocked_gpu_probes", lambda: [])
    xml = "<nvidia_smi_log>" + " " * 5000 + "<gpu><processes/></gpu></nvidia_smi_log>"

    def launch(*args, **kwargs):
        kwargs["stdout"].write(xml.encode())
        kwargs["stdout"].flush()
        return mock.Mock(wait=mock.Mock(return_value=0))

    with mock.patch.object(gpu_health.subprocess, "Popen", side_effect=launch):
        assert gpu_health.check_gpu_availability([4]).healthy
        xml = " " * (1024 * 1024 + 1)
        assert not gpu_health.check_gpu_availability([4]).healthy
