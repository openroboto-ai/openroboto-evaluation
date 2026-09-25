#!/usr/bin/env python3
"""Serve an AXIS deployment-canary checkpoint over the OpenPI wire protocol."""

from __future__ import annotations

import argparse
import json
import pathlib

from openpi_client import msgpack_numpy
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from axis_canary import AxisCanaryPolicy, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--chunk-size", type=int, default=5)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)

    def handler(websocket) -> None:
        policy = AxisCanaryPolicy(checkpoint, chunk_size=args.chunk_size)
        websocket.send(
            msgpack_numpy.packb({
                "checkpoint_sha256": checkpoint.metadata["checkpoint_sha256"],
                "deployment_canary_only": True,
                "task_id": checkpoint.metadata["task_id"],
            })
        )
        try:
            for message in websocket:
                request = msgpack_numpy.unpackb(message)
                response = policy.infer(request)
                websocket.send(msgpack_numpy.packb(response))
        except ConnectionClosed:
            return
        except Exception as exc:  # noqa: BLE001
            websocket.send(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))

    print(
        f"AXIS deployment canary listening on ws://{args.host}:{args.port} for task {checkpoint.metadata['task_id']}",
        flush=True,
    )
    with serve(handler, args.host, args.port, compression=None, max_size=None) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
