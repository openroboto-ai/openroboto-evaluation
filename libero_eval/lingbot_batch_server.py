"""Dynamic and deterministic static-batch servers for LingBot-VLA 2.0.

The upstream LingBot WebSocket handler calls the synchronous policy directly
from each connection handler.  That serializes requests at batch size one.
This server keeps the wire protocol unchanged, but queues action requests and
lets one inference worker drain up to ``max_batch`` requests at a time.

No request waits for a full batch by default.  While one batch is executing in
a worker thread, the asyncio event loop continues receiving the next batch.
Batch shapes are padded to powers of two so torch.compile sees a small, bounded
set of shapes.  Reset messages are serialized with inference but never batched.

The static server assigns every evaluator worker to a stable lane. Lanes are
split into fixed-size cohorts and every model call has exactly ``batch_size``
slots in lane order. Active lanes synchronize at an inference barrier; absent
tail lanes are filled with ignored duplicates. This keeps the compiled graph,
sample position, and cohort membership stable while two cohorts can still
overlap simulation with inference on one GPU.
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback
from dataclasses import dataclass

from deploy.msgpack_numpy import Packer, unpackb
import websockets.asyncio.server as _server
import websockets.frames

from lingbot_eval_protocol import (
    POLICY_BATCH_LANE_FIELD,
    POLICY_BATCH_LANE_RELEASE_FIELD,
    validate_policy_batch_lane,
)


logger = logging.getLogger(__name__)


def next_power_of_two(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"batch size must be a positive integer, got {value!r}")
    return 1 << (value - 1).bit_length()


@dataclass(slots=True)
class _PendingRequest:
    observation: dict
    future: asyncio.Future
    enqueued_at: float
    lane: int | None = None


@dataclass(slots=True)
class _LaneStateChange:
    lane: int
    active: bool


class LingbotDynamicBatchServer:
    """Protocol-compatible LingBot server with one greedy batch queue."""

    def __init__(
        self,
        policy,
        *,
        host: str,
        port: int,
        max_batch: int,
        batch_wait_ms: float = 0.0,
    ) -> None:
        if not isinstance(max_batch, int) or isinstance(max_batch, bool) or max_batch < 1:
            raise ValueError(f"max_batch must be a positive integer, got {max_batch!r}")
        if next_power_of_two(max_batch) != max_batch:
            raise ValueError(f"max_batch must be a power of two, got {max_batch}")
        if batch_wait_ms < 0:
            raise ValueError(f"batch_wait_ms must be non-negative, got {batch_wait_ms}")
        if not callable(getattr(policy, "infer_batch", None)):
            raise TypeError("LingBot dynamic batching requires policy.infer_batch")

        self._policy = policy
        self._host = host
        self._port = port
        self._max_batch = max_batch
        self._batch_wait_s = batch_wait_ms / 1000.0
        self._queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()
        self._policy_lock: asyncio.Lock | None = None

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        self._policy_lock = asyncio.Lock()
        worker = asyncio.create_task(self._infer_worker())
        try:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
            ) as server:
                await server.serve_forever()
        finally:
            worker.cancel()

    async def _infer_worker(self) -> None:
        while True:
            items = [await self._queue.get()]
            if self._batch_wait_s:
                await asyncio.sleep(self._batch_wait_s)
            while len(items) < self._max_batch:
                try:
                    items.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            real_batch_size = len(items)
            padded_to = next_power_of_two(real_batch_size)
            observations = [item.observation for item in items]
            observations.extend([observations[-1]] * (padded_to - real_batch_size))
            started = time.monotonic()
            try:
                assert self._policy_lock is not None
                async with self._policy_lock:
                    padded_results = await asyncio.to_thread(self._policy.infer_batch, observations)
                infer_ms = (time.monotonic() - started) * 1000
                if len(padded_results) != padded_to:
                    raise RuntimeError(
                        f"LingBot batch returned {len(padded_results)} results for padded batch {padded_to}"
                    )
                for item, result in zip(items, padded_results[:real_batch_size]):
                    result = dict(result)
                    result["server_timing"] = {
                        "infer_ms": infer_ms,
                        "batch_size": real_batch_size,
                        "padded_to": padded_to,
                        "queue_ms": (started - item.enqueued_at) * 1000,
                    }
                    if not item.future.done():
                        item.future.set_result(result)
            except Exception as exc:  # noqa: BLE001 - forwarded to each waiting client
                for item in items:
                    if not item.future.done():
                        item.future.set_exception(exc)

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack({}))
        while True:
            try:
                observation = unpackb(await websocket.recv())
                if isinstance(observation, dict) and observation.get("reset"):
                    started = time.monotonic()
                    assert self._policy_lock is not None
                    async with self._policy_lock:
                        result = await asyncio.to_thread(self._policy.infer, observation)
                    result = dict(result)
                    result["server_timing"] = {
                        "infer_ms": (time.monotonic() - started) * 1000,
                        "batch_size": 1,
                        "padded_to": 1,
                        "queue_ms": 0.0,
                    }
                else:
                    future = asyncio.get_running_loop().create_future()
                    await self._queue.put(
                        _PendingRequest(
                            observation=observation,
                            future=future,
                            enqueued_at=time.monotonic(),
                        )
                    )
                    result = await future
                await websocket.send(packer.pack(result))
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class LingbotStaticBatchServer(LingbotDynamicBatchServer):
    """Fixed-cohort server used by the reproducible LingBot evaluation path."""

    def __init__(
        self,
        policy,
        *,
        host: str,
        port: int,
        batch_size: int,
        lane_count: int,
    ) -> None:
        super().__init__(policy, host=host, port=port, max_batch=batch_size)
        if lane_count < batch_size or lane_count % batch_size:
            raise ValueError(
                f"static lane_count must be a positive multiple of batch_size, got "
                f"lane_count={lane_count} batch_size={batch_size}"
            )
        self._batch_size = batch_size
        self._lane_count = lane_count
        self._events: asyncio.Queue[_PendingRequest | _LaneStateChange] = asyncio.Queue()
        self._lane_connections: dict[int, _server.ServerConnection] = {}

    def _cohort(self, lane: int) -> int:
        if lane >= self._lane_count:
            raise ValueError(f"evaluation batch lane {lane} is outside configured lane_count={self._lane_count}")
        return lane // self._batch_size

    async def _infer_worker(self) -> None:
        # run_eval guarantees that every configured lane has at least one task.
        # Treat them all as active before clients finish importing LIBERO so no
        # startup-timing race can create an under-filled first model call.
        active: set[int] = set(range(self._lane_count))
        pending: dict[int, _PendingRequest] = {}
        cohort_count = self._lane_count // self._batch_size
        next_cohort = 0

        def apply_event(event: _PendingRequest | _LaneStateChange) -> None:
            if isinstance(event, _LaneStateChange):
                if event.active:
                    active.add(event.lane)
                else:
                    active.discard(event.lane)
                    abandoned = pending.pop(event.lane, None)
                    if abandoned is not None and not abandoned.future.done():
                        abandoned.future.set_exception(
                            RuntimeError(f"static batch lane {event.lane} was released with a request pending")
                        )
                return
            assert event.lane is not None
            if event.lane in pending:
                previous = pending[event.lane]
                if not previous.future.done():
                    if not event.future.done():
                        event.future.set_exception(
                            RuntimeError(f"static batch lane {event.lane} already has a request")
                        )
                    return
            pending[event.lane] = event

        def cohort_ready(cohort: int) -> bool:
            first = cohort * self._batch_size
            cohort_active = {lane for lane in active if first <= lane < first + self._batch_size}
            return bool(cohort_active) and cohort_active.issubset(pending)

        while True:
            # Always apply every event already received before choosing a ready
            # cohort. In particular, a release must not lose a race against a
            # batch that became ready one event earlier.
            while True:
                try:
                    apply_event(self._events.get_nowait())
                except asyncio.QueueEmpty:
                    break

            ready = None
            for offset in range(cohort_count):
                cohort = (next_cohort + offset) % cohort_count
                if cohort_ready(cohort):
                    ready = cohort
                    break
            if ready is None:
                apply_event(await self._events.get())
                continue

            next_cohort = (ready + 1) % cohort_count
            first = ready * self._batch_size
            cohort_lanes = range(first, first + self._batch_size)
            real_items = {lane: pending.pop(lane) for lane in cohort_lanes if lane in active}
            # cohort_ready guarantees at least one real request. Fill inactive
            # tail slots with a stable-shaped duplicate and ignore its output.
            template = real_items[min(real_items)].observation
            observations = []
            for lane in cohort_lanes:
                item = real_items.get(lane)
                observations.append(item.observation if item is not None else dict(template))

            started = time.monotonic()
            try:
                assert self._policy_lock is not None
                async with self._policy_lock:
                    batch_results = await asyncio.to_thread(self._policy.infer_batch, observations)
                infer_ms = (time.monotonic() - started) * 1000
                if len(batch_results) != self._batch_size:
                    raise RuntimeError(
                        f"LingBot static batch returned {len(batch_results)} results for batch {self._batch_size}"
                    )
                for lane, item in real_items.items():
                    result = dict(batch_results[lane - first])
                    result["server_timing"] = {
                        "infer_ms": infer_ms,
                        "batch_size": len(real_items),
                        "padded_to": self._batch_size,
                        "queue_ms": (started - item.enqueued_at) * 1000,
                        "batch_mode": "static",
                        "cohort": ready,
                        "lane": lane,
                    }
                    if not item.future.done():
                        item.future.set_result(result)
            except Exception as exc:  # noqa: BLE001 - forwarded to every cohort client
                for item in real_items.values():
                    if not item.future.done():
                        item.future.set_exception(exc)

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack({}))
        lane = None
        try:
            while True:
                observation = unpackb(await websocket.recv())
                if not isinstance(observation, dict):
                    raise ValueError("LingBot static batching expects an observation mapping")
                if POLICY_BATCH_LANE_FIELD not in observation:
                    raise ValueError(f"LingBot static batching requires {POLICY_BATCH_LANE_FIELD!r}")
                request_lane = validate_policy_batch_lane(observation.pop(POLICY_BATCH_LANE_FIELD))
                self._cohort(request_lane)
                if observation.pop(POLICY_BATCH_LANE_RELEASE_FIELD, False) is True:
                    if set(observation) != set():
                        raise ValueError("static batch lane release request contains unexpected fields")
                    await self._events.put(_LaneStateChange(request_lane, False))
                    await websocket.send(packer.pack({"released_lane": request_lane}))
                    return
                if lane is None:
                    lane = request_lane
                    previous = self._lane_connections.get(lane)
                    self._lane_connections[lane] = websocket
                    if previous is None:
                        # Normally redundant because configured lanes start
                        # active; required when a lane reconnects after an
                        # explicit release in a diagnostic run.
                        await self._events.put(_LaneStateChange(lane, True))
                elif request_lane != lane:
                    raise ValueError(f"connection changed static batch lane from {lane} to {request_lane}")

                if observation.get("reset"):
                    started = time.monotonic()
                    assert self._policy_lock is not None
                    async with self._policy_lock:
                        result = await asyncio.to_thread(self._policy.infer, observation)
                    result = dict(result)
                    result["server_timing"] = {
                        "infer_ms": (time.monotonic() - started) * 1000,
                        "batch_size": 1,
                        "padded_to": 1,
                        "queue_ms": 0.0,
                        "batch_mode": "reset",
                        "lane": lane,
                    }
                else:
                    future = asyncio.get_running_loop().create_future()
                    await self._events.put(
                        _PendingRequest(
                            observation=observation,
                            future=future,
                            enqueued_at=time.monotonic(),
                            lane=lane,
                        )
                    )
                    result = await future
                await websocket.send(packer.pack(result))
        except websockets.ConnectionClosed:
            logger.info("Connection from %s closed", websocket.remote_address)
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Internal server error. Traceback included in previous frame.",
            )
            raise
        finally:
            if lane is not None and self._lane_connections.get(lane) is websocket:
                self._lane_connections.pop(lane, None)
                # A lane spans several short-lived eval_task processes. A TCP
                # disconnect is therefore only a handoff, never proof that the
                # lane is exhausted. worker_loop sends an explicit release once
                # its deterministic task queue is empty.
