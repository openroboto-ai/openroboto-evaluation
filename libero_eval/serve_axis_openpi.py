#!/usr/bin/env python3
"""Serve a native-joint Pi0.5 AXIS checkpoint without patching OpenPI."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import pathlib
import socket
import sys


VALIDATOR_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VALIDATOR_ROOT / "libero_eval"))

_TASK_ID_KEY = "_axis_policy_task_id"
_TRIAL_KEY = "_axis_policy_trial"
_CALL_KEY = "_axis_policy_call"


class DeterministicAxisPolicy:
    """Supply request-addressed diffusion noise independent of server ordering."""

    def __init__(
        self,
        policy,
        *,
        seed: int,
        action_horizon: int,
        action_dim: int,
        policy_samples: int = 1,
        sample_reduction: str = "mean",
    ) -> None:
        if type(policy_samples) is not int or not 1 <= policy_samples <= 16:
            raise ValueError("policy_samples must be an integer between 1 and 16")
        if sample_reduction not in ("mean", "medoid"):
            raise ValueError("sample_reduction must be mean or medoid")
        self._policy = policy
        self._seed = int(seed)
        self._action_horizon = int(action_horizon)
        self._action_dim = int(action_dim)
        self._policy_samples = policy_samples
        self._sample_reduction = sample_reduction

    @property
    def metadata(self) -> dict:
        return {
            **self._policy.metadata,
            "axis_policy_seed": self._seed,
            "axis_policy_samples": self._policy_samples,
            "axis_sample_reduction": self._sample_reduction,
            "axis_noise_derivation": (
                "sha256(seed,task_id,trial,inference_call)+numpy-pcg64-v1; "
                "sample 0 uses the legacy address, additional samples append ':sample:N'"
            ),
        }

    def infer(self, observation: dict) -> dict:
        import numpy as np

        cleaned = dict(observation)
        try:
            coordinates = tuple(int(cleaned.pop(key)) for key in (_TASK_ID_KEY, _TRIAL_KEY, _CALL_KEY))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("AXIS policy request is missing deterministic task/trial/call coordinates") from exc
        material = ":".join(str(value) for value in (self._seed, *coordinates)).encode("ascii")
        predictions = []
        for member in range(self._policy_samples):
            addressed = material if member == 0 else material + f":sample:{member}".encode("ascii")
            noise_seed = int.from_bytes(hashlib.sha256(addressed).digest()[:8], "big")
            noise = np.random.default_rng(noise_seed).standard_normal(
                (self._action_horizon, self._action_dim), dtype=np.float32
            )
            predictions.append(self._policy.infer(cleaned, noise=noise))
        if self._policy_samples == 1:
            return predictions[0]
        result = dict(predictions[0])
        # Uniform policy ensemble; neither checker feedback nor task-specific
        # action templates enter the reduction. Every member uses the same weights.
        chunks = np.stack([value["actions"] for value in predictions])
        center = np.mean(chunks, axis=0, dtype=np.float64)
        if self._sample_reduction == "mean":
            result["actions"] = center
        else:
            # A squared-Euclidean medoid preserves one whole sampled chunk,
            # including its gripper/arm timing. Metric: native 9D joint targets,
            # summed across the entire horizon, with no task-dependent weights.
            distances = np.sum((chunks - center) ** 2, axis=(1, 2))
            result["actions"] = chunks[int(np.argmin(distances))]
        result["policy_timing"] = {
            "infer_ms": sum(value.get("policy_timing", {}).get("infer_ms", 0.0) for value in predictions)
        }
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--config", default="pi05_axis_joint", choices=("pi05_axis_joint",))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gripper-mode", choices=("continuous", "symmetric-binary"), default="continuous")
    parser.add_argument("--policy-samples", type=int, choices=range(1, 17), default=1)
    parser.add_argument("--sample-reduction", choices=("mean", "medoid"), default="mean")
    parser.add_argument(
        "--openpi-root",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("OPENPI_DIR", VALIDATOR_ROOT / "third_party" / "openpi")),
    )
    args = parser.parse_args()
    if args.policy_samples > 1 and args.gripper_mode != "continuous":
        parser.error("multi-sample policy currently requires continuous gripper decoding")

    from axis_model_input import AXIS_PI05_DISCRETE_STATE_INPUT

    checkpoint = args.checkpoint.expanduser().resolve()

    from axis_openpi_sources import bind_openpi_sources

    bind_openpi_sources(args.openpi_root)

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server

    from axis_openpi_config import make_config

    config = make_config(
        gripper_mode=args.gripper_mode,
        discrete_state_input=AXIS_PI05_DISCRETE_STATE_INPUT,
    )
    policy = DeterministicAxisPolicy(
        policy_config.create_trained_policy(config, checkpoint),
        seed=args.seed,
        action_horizon=config.model.action_horizon,
        action_dim=config.model.action_dim,
        policy_samples=args.policy_samples,
        sample_reduction=args.sample_reduction,
    )
    hostname = socket.gethostname()
    logging.info("Creating AXIS policy server (host: %s, port: %d, seed: %d)", hostname, args.port, args.seed)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
