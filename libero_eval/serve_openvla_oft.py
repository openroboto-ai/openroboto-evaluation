"""Serve an OpenVLA-OFT LIBERO checkpoint over the validator websocket API.

The model stays in its upstream checkpoint format.  This adapter only translates
the validator's observation schema to the official OpenVLA-OFT inference path and
applies the gripper post-processing used by run_libero_eval.py.
"""

from __future__ import annotations

import argparse
import asyncio
import http
import json
import logging
import os
import pathlib
import random
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
import websockets
import websockets.asyncio.server
import websockets.frames

from openpi_client import msgpack_numpy


def _load_policy(checkpoint: pathlib.Path):
    # Imports are intentionally delayed until after CLI parsing.  Upstream chooses
    # LIBERO constants by looking for "libero" in sys.argv.
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_processor,
        get_proprio_projector,
        get_vla_action,
    )
    from transformers import AutoModelForVision2Seq

    cfg = SimpleNamespace(
        pretrained_checkpoint=str(checkpoint),
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        load_in_8bit=False,
        load_in_4bit=False,
        unnorm_key="libero_spatial",
    )
    logging.info("loading OpenVLA-OFT checkpoint %s", checkpoint)
    model = AutoModelForVision2Seq.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")
    model.vision_backbone.set_num_images_in_input(2)
    model.eval()
    model.norm_stats = json.loads((checkpoint / "dataset_statistics.json").read_text())
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, model.llm_dim)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    def infer(observation: dict) -> dict:
        suite = str(observation.get("task_suite", "libero_spatial"))
        if suite not in model.norm_stats:
            raise ValueError(f"checkpoint has no normalization statistics for suite {suite!r}")
        cfg.unnorm_key = suite
        obs = {
            # eval_task sends the original rotated 256x256 render for this family;
            # upstream get_vla_action performs JPEG/Lanczos resize + center crop.
            "full_image": np.asarray(observation["observation/image"]),
            "wrist_image": np.asarray(observation["observation/wrist_image"]),
            "state": np.asarray(observation["observation/state"]),
        }
        actions = np.asarray(
            get_vla_action(
                cfg,
                model,
                processor,
                obs,
                str(observation["prompt"]),
                action_head=action_head,
                proprio_projector=proprio_projector,
                use_film=False,
            )
        )
        # Official process_action(): [0,1] -> [-1,+1], binarize, then invert
        # because the OpenVLA RLDS loader uses the opposite gripper convention.
        actions[..., -1] = -np.sign(2.0 * actions[..., -1] - 1.0)
        return {"actions": actions}

    return infer


async def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def _serve(infer, host: str, port: int) -> None:
    async def handler(websocket):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack({"model_family": "openvla_oft", "action_horizon": 8}))
        while True:
            try:
                observation = msgpack_numpy.unpackb(await websocket.recv())
                started = time.monotonic()
                result = infer(observation)
                result["server_timing"] = {
                    "infer_ms": (time.monotonic() - started) * 1000.0,
                    "batch_size": 1,
                }
                await websocket.send(packer.pack(result))
            except websockets.ConnectionClosed:
                return
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="OpenVLA-OFT inference error; traceback sent in previous frame",
                )
                logging.exception("inference request failed")
                return

    async with websockets.asyncio.server.serve(
        handler,
        host,
        port,
        compression=None,
        max_size=None,
        process_request=_health_check,
    ) as server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # run_eval's readiness probe opens and closes a raw TCP socket.  websockets
    # 16 logs that harmless probe as an InvalidMessage traceback unless muted.
    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(int(os.environ.get("OPENVLA_TORCH_THREADS", "2")))
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    infer = _load_policy(args.checkpoint.resolve())
    logging.info("OpenVLA-OFT policy ready on %s:%d", args.host, args.port)
    asyncio.run(_serve(infer, args.host, args.port))


if __name__ == "__main__":
    main()
