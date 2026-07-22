"""openpi policy server with greedy dynamic batching.

Drop-in replacement for openpi's scripts/serve_policy.py (same websocket
protocol, same client), designed for many concurrent eval clients per GPU:

  - Each connection enqueues its observation and awaits the result.
  - A single inference worker drains WHATEVER is currently queued (1..max_batch
    requests), runs them as ONE batched sample_actions call, and replies.

Nobody waits for a batch to fill: when requests are sparse the effective batch
is 1 and latency matches the upstream server; when the GPU is the bottleneck
the queue is naturally non-empty and batches form by themselves (throughput
rises exactly when it is needed). pi0.5 inference is memory-bound at batch 1,
so a batch of 4 costs far less than 4x.

Batches are padded to the next power of two (repeating the last sample) so JAX
only ever compiles log2(max_batch)+1 shapes; combined with
JAX_COMPILATION_CACHE_DIR those compilations happen once ever.

This file lives in the validator repo but runs inside the openpi server venv
(third_party is read-only per project policy). It reaches into openpi's
Policy internals (_input_transform / _sample_actions / _output_transform /
_rng / _sample_kwargs) — the seams openpi does not expose separately. An
openpi upgrade that renames them fails loudly at startup, not with wrong
results.

Run (from run_eval.py or manually):
    <openpi-venv>/python serve_policy_batched.py \
        --port 9000 --config pi05_libero --dir <ckpt> --max-batch 8
"""

import argparse
import asyncio
import http
import logging
import time
import traceback

import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


class BatchedPolicy:
    """Wraps an openpi Policy to run a list of observations as one batch.

    Reproduces Policy.infer() exactly (see openpi policies/policy.py), with the
    single-sample `[np.newaxis]` batch dim replaced by a stack over requests.
    """

    def __init__(self, policy, max_batch: int):
        import jax  # noqa: PLC0415  (imported here: this venv-only module must not import at parse time elsewhere)

        self._policy = policy
        self._max_batch = max_batch
        self._jax = jax
        # Fail loudly now (not mid-eval) if openpi's Policy internals changed.
        for attr in ("_input_transform", "_output_transform", "_sample_actions", "_sample_kwargs", "_is_pytorch_model"):
            if not hasattr(policy, attr):
                raise AttributeError(f"openpi Policy no longer has {attr}; update serve_policy_batched.py")
        if policy._is_pytorch_model and max_batch > 1:
            # pi0_pytorch wraps sample_actions in torch.compile, which recompiles
            # for every new input shape: a dynamic batch cycling through
            # {1,2,4,8} recompiles endlessly (measured 7x SLOWER per sample).
            # Until the torch path pre-warms fixed shapes, batch=1 keeps the
            # compiled shape unique; multi-worker overlap still applies.
            logger.warning("PyTorch checkpoint: forcing max_batch=1 (torch.compile is shape-static)")
            self._max_batch = 1

    @property
    def metadata(self) -> dict:
        return self._policy.metadata

    def infer_batch(self, obs_list: list[dict]) -> list[dict]:
        import jax.numpy as jnp  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from openpi.models import model as _model  # noqa: PLC0415

        jax = self._jax
        policy = self._policy

        # Per-sample input transforms (CPU: resize, normalize, tokenize).
        inputs = [policy._input_transform(jax.tree.map(lambda x: x, o)) for o in obs_list]

        b = len(inputs)
        padded = next_pow2(b)
        stacked_list = inputs + [inputs[-1]] * (padded - b)

        if policy._is_pytorch_model:
            import torch  # noqa: PLC0415

            device = policy._pytorch_device
            batch = jax.tree.map(
                lambda *xs: torch.stack([torch.from_numpy(np.array(x)) for x in xs]).to(device), *stacked_list
            )
            rng_or_device = device
        else:
            batch = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *stacked_list)
            policy._rng, rng_or_device = jax.random.split(policy._rng)

        observation = _model.Observation.from_dict(batch)
        start = time.monotonic()
        actions = policy._sample_actions(rng_or_device, observation, **policy._sample_kwargs)
        model_ms = (time.monotonic() - start) * 1000

        results = []
        for i in range(b):
            if policy._is_pytorch_model:
                out = {"state": batch["state"][i], "actions": actions[i]}
                out = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), out)
            else:
                out = {"state": batch["state"][i], "actions": actions[i]}
                out = jax.tree.map(np.asarray, out)
            out = policy._output_transform(out)
            out["policy_timing"] = {"infer_ms": model_ms, "batch_size": b, "padded_to": padded}
            results.append(out)
        return results


class BatchedPolicyServer:
    """Websocket server, protocol-compatible with openpi's WebsocketPolicyServer."""

    def __init__(self, policy: BatchedPolicy, host: str, port: int) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._queue: asyncio.Queue = asyncio.Queue()

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
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
        """Drain whatever is queued (1..max_batch), run one batched inference, reply.

        Inference runs in a thread so the event loop keeps accepting requests —
        the NEXT batch accumulates while the current one is on the GPU.
        """
        while True:
            items = [await self._queue.get()]
            while len(items) < self._policy._max_batch:
                try:
                    items.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            obs_list = [obs for obs, _ in items]
            try:
                results = await asyncio.to_thread(self._policy.infer_batch, obs_list)
                for (_, (future, t_enq)), result in zip(items, results):
                    result.setdefault("server_timing", {})
                    result["server_timing"]["infer_ms"] = result["policy_timing"]["infer_ms"]
                    result["server_timing"]["batch_size"] = result["policy_timing"]["batch_size"]
                    result["server_timing"]["queue_ms"] = (time.monotonic() - t_enq) * 1000
                    if not future.done():
                        future.set_result(result)
            except Exception as e:  # noqa: BLE001
                for _, (future, _t) in items:
                    if not future.done():
                        future.set_exception(e)

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        from openpi_client import msgpack_numpy  # noqa: PLC0415

        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._policy.metadata))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                future: asyncio.Future = asyncio.get_running_loop().create_future()
                await self._queue.put((obs, (future, time.monotonic())))
                action = await future
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config", required=True, help="openpi training config name (e.g. pi05_libero)")
    parser.add_argument("--dir", required=True, help="checkpoint directory")
    parser.add_argument("--max-batch", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)

    from openpi.policies import policy_config as _policy_config  # noqa: PLC0415
    from openpi.training import config as _config  # noqa: PLC0415

    logger.info(f"Loading policy: config={args.config} dir={args.dir}")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config), args.dir)
    batched = BatchedPolicy(policy, max_batch=args.max_batch)
    logger.info(f"Serving on localhost:{args.port} (max_batch={args.max_batch})")
    BatchedPolicyServer(batched, host="localhost", port=args.port).serve_forever()


if __name__ == "__main__":
    main()
