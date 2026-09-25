"""CPU-only GPU probes with bounded cleanup, including unkillable D-state children."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re
import signal
import subprocess
import tempfile
import xml.etree.ElementTree as ET

GPU_CHECK_INTERVAL = 60
GPU_PROBE_TIMEOUT = 15


@dataclasses.dataclass(frozen=True)
class GpuHealth:
    healthy: bool
    detail: str


def blocked_gpu_probes(proc_root: pathlib.Path = pathlib.Path("/proc")) -> list[int]:
    """Avoid accumulating more probes while an earlier one is stuck in the driver."""
    blocked = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = dict(line.split(":", 1) for line in (entry / "status").read_text().splitlines())
        except (OSError, ValueError):
            continue  # Processes can exit or belong to a different user.
        if fields.get("Name", "").strip() == "nvidia-smi" and fields.get("State", "").strip().startswith("D"):
            blocked.append(int(entry.name))
    return sorted(blocked)


def _probe_gpu(arguments: list[str], *, timeout: float) -> GpuHealth:
    if timeout <= 0:
        raise ValueError("GPU probe timeout must be positive")
    blocked = blocked_gpu_probes()
    if blocked:
        return GpuHealth(False, f"nvidia-smi stuck in uninterruptible D state; pids={blocked}")
    # Do not use subprocess.run or a Popen context manager: both wait without a
    # deadline after kill. A child blocked in the NVIDIA kernel driver cannot exit.
    with tempfile.TemporaryFile() as output:
        try:
            proc = subprocess.Popen(
                ["nvidia-smi", *arguments],
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            return GpuHealth(False, f"cannot start nvidia-smi ({type(exc).__name__})")
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            return GpuHealth(False, f"nvidia-smi timed out after {timeout:g}s; pid={proc.pid}")
        output.seek(0)
        # XML includes graphics processes as well as CUDA processes and can be
        # much larger than the short device list. Fail closed on truncation.
        raw = output.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            return GpuHealth(False, "nvidia-smi output exceeded 1 MiB")
        detail = raw.decode("utf-8", errors="replace").strip()
    if returncode != 0:
        return GpuHealth(False, f"nvidia-smi exited {returncode}: {detail[:4096]}")
    return GpuHealth(True, detail)


def check_gpu_health(*, timeout: float = GPU_PROBE_TIMEOUT) -> GpuHealth:
    probe = _probe_gpu(["-L"], timeout=timeout)
    if not probe.healthy:
        return probe
    detail = probe.detail
    if not re.search(r"^GPU \d+:", detail, flags=re.MULTILINE):
        return GpuHealth(False, f"nvidia-smi returned no GPUs: {detail}")
    if re.search(r"error|failed|unknown", detail, flags=re.IGNORECASE):
        return GpuHealth(False, f"nvidia-smi reported a GPU error: {detail}")
    return GpuHealth(True, detail)


def check_gpu_availability(gpus: list[int], *, timeout: float = GPU_PROBE_TIMEOUT) -> GpuHealth:
    """Require idle selected GPUs before launching a self-contained evaluation.

    Advisory file locks only coordinate cooperating evaluators. Inspect both
    compute and graphics clients so an existing training job, renderer, or
    leftover evaluator prevents us from starting on the same device. This is
    a snapshot, not an exclusive reservation against future external jobs.
    """
    if not gpus or len(gpus) != len(set(gpus)) or any(gpu < 0 for gpu in gpus):
        raise ValueError("GPU ids must be nonnegative, unique, and nonempty")
    probe = _probe_gpu(["-i", ",".join(map(str, gpus)), "-q", "-x"], timeout=timeout)
    if not probe.healthy:
        return probe
    try:
        root = ET.fromstring(probe.detail)
    except ET.ParseError:
        return GpuHealth(False, "cannot parse nvidia-smi GPU occupancy XML")
    devices = root.findall("gpu")
    if len(devices) != len(gpus):
        return GpuHealth(False, f"nvidia-smi returned {len(devices)} devices for selected GPUs {gpus}")
    busy = []
    for device in devices:
        identity = device.get("id", "unknown PCI device")
        processes = device.find("processes")
        if processes is None or (processes.text or "").strip():
            return GpuHealth(False, f"GPU {identity}: process accounting unavailable")
        for process in processes:
            pid = process.findtext("pid", "").strip()
            if process.tag != "process_info" or not pid.isdigit() or int(pid) <= 0:
                return GpuHealth(False, f"GPU {identity}: invalid process accounting")
            busy.append(f"GPU {identity}: pid={pid} type={process.findtext('type', 'unknown')}")
    if busy:
        return GpuHealth(False, "selected GPUs already in use; " + "; ".join(busy))
    return GpuHealth(True, f"selected GPUs {gpus} have no active compute or graphics processes")
