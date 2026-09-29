"""Periodic AXIS progress events consumed by benchmark_worker.

Only completed task results contribute to the episode counter. Intermediate
trials stay in their task process until that process returns its result.
"""

from __future__ import annotations

import contextlib
import json
import logging
import pathlib
import threading
from collections.abc import Callable, Iterator, Sequence, Mapping
from typing import Any


PROGRESS_INTERVAL_S = 30.0
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def report_axis_progress(
    path: pathlib.Path | None,
    benchmark: str,
    task_ids: Sequence[int],
    trials_per_task: int | Mapping[int, int],
    *,
    interval_s: float = PROGRESS_INTERVAL_S,
) -> Iterator[Callable[[int, dict[str, Any]], None]]:
    """Emit at start, each task completion, every interval, and shutdown.

    The single writer lock protects both the snapshot and JSONL writes. HTTP
    delivery remains in benchmark_worker, outside the evaluation processes.
    """
    if path is None:
        yield lambda task_id, result: None
        return
    if interval_s <= 0:
        raise ValueError("AXIS progress interval must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = set(task_ids)
    completed: dict[int, dict[str, Any]] = {}
    detail: dict[str, Any] = {
        "benchmark": benchmark,
        "tasks_done": 0,
        "tasks_total": len(task_ids),
        "tasks_failed": 0,
        "episodes_done": 0,
        "episodes_total": sum(trials_per_task[tid] for tid in task_ids)
        if isinstance(trials_per_task, Mapping)
        else len(task_ids) * trials_per_task,
    }
    lock = threading.Lock()
    stopped = threading.Event()

    def emit_locked() -> None:
        try:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"stage": "evaluating", "detail": detail}) + "\n")
        except OSError:
            # Progress I/O must not discard an expensive evaluation result.
            logger.warning("Could not write AXIS progress to %s", path, exc_info=True)

    def record(task_id: int, result: dict[str, Any]) -> None:
        if task_id not in selected:
            raise ValueError(f"AXIS progress received an unselected task: {task_id}")
        with lock:
            completed[task_id] = result
            detail.update(
                tasks_done=len(completed),
                tasks_failed=sum(item.get("status") != "ok" for item in completed.values()),
                episodes_done=sum(int(item.get("num_trials", 0)) for item in completed.values()),
                last_completed_task_id=str(task_id),
            )
            emit_locked()

    def heartbeat() -> None:
        while not stopped.wait(interval_s):
            with lock:
                emit_locked()

    with lock:
        emit_locked()
    thread = threading.Thread(target=heartbeat, name="axis-progress", daemon=True)
    thread.start()
    try:
        yield record
    finally:
        stopped.set()
        thread.join()
        with lock:
            emit_locked()
