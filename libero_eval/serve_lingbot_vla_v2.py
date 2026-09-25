"""Validator-owned LingBot-VLA 2.0 server with an explicit robot contract."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

import numpy as np
import torch
import yaml
from transformers import AutoConfig

from deploy.lingbot_vla_v2_policy import LingBotVlaV2InferencePolicy, LingbotVLAv2Server, str2bool
from deploy.websocket_policy_server import WebsocketPolicyServer
from lingbotvla.data.vla_data.utils import FeatureTransform
from lingbotvla.models import build_processor
from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config

from lingbot_runtime import (
    LINGBOT_ACTION_REPLAY_ATOL,
    LINGBOT_TORCH_INTEROP_THREADS,
    LINGBOT_TORCH_THREADS,
    LingbotRequestSeededSampler,
    copy_readonly_observation_arrays,
    disable_lingbot_batch1_vision_cache,
    lingbot_actions_are_equivalent,
    load_data_contract,
    verify_lingbot_request_seed_determinism_nonblocking,
    warm_up_lingbot_libero_policy,
)
from lingbot_eval_protocol import POLICY_RNG_FIELD, validate_policy_seed


class EvaluatorLingbotServer(LingbotVLAv2Server):
    def __init__(
        self,
        *args,
        robot_config_root: pathlib.Path,
        data_contract: pathlib.Path,
        **kwargs,
    ):
        self._robot_config_root = robot_config_root
        self._data_contract = data_contract
        super().__init__(*args, **kwargs)

    def _prepare_model_input(self, observation):
        """Detach decoded arrays from immutable WebSocket message buffers."""
        return super()._prepare_model_input(copy_readonly_observation_arrays(observation))

    def load_vla(self, path_to_pi_model):
        """Load a standard HF checkpoint without upstream's fixed directory layout.

        Upstream reconstructs the architecture from
        ``../../../lingbotvla_cli.yaml``. A validator submission is a normal
        Hugging Face checkpoint whose ``config.json`` already carries those
        architecture fields, while its data mapping is evaluator-owned.
        """
        checkpoint = pathlib.Path(path_to_pi_model)
        config_path = checkpoint / "config.json"
        try:
            config_payload = json.loads(config_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load LingBot checkpoint config {config_path}: {exc}") from exc
        if not isinstance(config_payload, dict):
            raise ValueError(f"LingBot checkpoint config must be an object: {config_path}")

        base_model_path = os.environ.get("QWEN3VL_PATH") or config_payload.get("tokenizer_path")
        if not isinstance(base_model_path, str) or not base_model_path:
            raise ValueError("QWEN3VL_PATH or config.json tokenizer_path is required")

        config = LingbotVLAV2Config(**config_payload)
        config.attention_implementation = "eager"
        config.tokenizer_path = base_model_path
        config.use_cache = True
        self.model_name = "qwen3vl"
        self.config = config
        self.merge_qwen_config(AutoConfig.from_pretrained(base_model_path))

        self.processor = build_processor(base_model_path)
        self.language_tokenizer = self.processor.tokenizer
        self.data_config = load_data_contract(self._data_contract)

        self.vla = LingBotVlaV2InferencePolicy(self.config, eval=True)
        self.load_model_weights(str(checkpoint), strict=True)
        self.vla.feature_transform = None
        self.vla.model._use_compile_predict_velocity = bool(self.use_compile)
        self.vla.model._compiled_predict_velocity = None
        sample_actions_fn = self.vla.model.sample_actions
        if self.use_compile:
            self.vla.model.qwenvl_with_expert = torch.compile(self.vla.model.qwenvl_with_expert)
            sample_actions_fn = torch.compile(self.vla.model.sample_actions)
        self._request_seeded_sampler = LingbotRequestSeededSampler(
            sample_actions_fn,
            generator_factory=torch.Generator,
            randn=torch.randn,
            concatenate=torch.cat,
            action_steps=self.config.n_action_steps,
            action_dimension=self.config.max_action_dim,
        )
        self.sample_actions_fn = self._request_seeded_sampler
        return self.vla

    def reset(self, robo_name, path_to_pi_model=None):
        if path_to_pi_model is not None:
            self.vla = self.load_vla(path_to_pi_model)
            if self.use_bf16:
                self.vla = self.vla.to(torch.bfloat16).cuda().eval()
            else:
                self.vla.model.float()
                self.vla = self.vla.cuda().eval()

        self.global_step = 0
        self.last_action_chunk = None
        self.last_normalized_action_chunk = None
        robot_config = self._robot_config_root / f"{robo_name}.yaml"
        if not robot_config.is_file():
            raise FileNotFoundError(f"unknown evaluator robot config {robo_name!r}: {robot_config}")
        # Parse once here for an early, local error; FeatureTransform remains
        # the source of truth for the mapping semantics.
        parsed = yaml.safe_load(robot_config.read_text())
        if not isinstance(parsed, dict):
            raise ValueError(f"robot config must be a mapping: {robot_config}")
        feature_transform = FeatureTransform(
            str(robot_config),
            self.data_config,
            self.config,
            self.processor,
            chunk_size=self.config.chunk_size,
            norm_stats_path=self.robot_norm_path,
        )
        self.vla.feature_transform = feature_transform
        self.action_key = feature_transform.org_features["actions"]

    def infer(self, observation, center_crop=True, return_normalized=False):
        """Run scored requests with RNG isolated from concurrent client ordering."""
        if not isinstance(observation, dict) or POLICY_RNG_FIELD not in observation:
            return super().infer(observation, center_crop=center_crop, return_normalized=return_normalized)

        observation = dict(observation)
        evaluation_seed = validate_policy_seed(observation.pop(POLICY_RNG_FIELD))
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all()
        try:
            random.seed(evaluation_seed)
            np.random.seed(evaluation_seed % (1 << 32))
            torch.manual_seed(evaluation_seed)
            torch.cuda.manual_seed_all(evaluation_seed)
            with self._request_seeded_sampler.request_seed(evaluation_seed):
                return super().infer(observation, center_crop=center_crop, return_normalized=return_normalized)
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.set_rng_state(torch_rng_state)
            torch.cuda.set_rng_state_all(cuda_rng_states)

    def infer_batch(self, observations):
        """Run a batch while preserving the independent seed of every request."""
        if not isinstance(observations, (list, tuple)) or not observations:
            raise ValueError("LingBot inference batch must be a non-empty list")

        seeds = []
        unseeded_observations = []
        for index, observation in enumerate(observations):
            if not isinstance(observation, dict) or observation.get("reset"):
                raise ValueError(f"LingBot batch item {index} must be an action observation")
            if POLICY_RNG_FIELD not in observation:
                raise ValueError(f"LingBot batch item {index} is missing {POLICY_RNG_FIELD}")
            observation = copy_readonly_observation_arrays(observation)
            seeds.append(validate_policy_seed(observation.pop(POLICY_RNG_FIELD)))
            unseeded_observations.append(observation)

        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all()
        try:
            # FeatureTransform is deterministic in policy_eval mode.  Seed the
            # remaining process-global RNGs predictably and inject each action
            # noise tensor independently through request_seeds below.
            random.seed(seeds[0])
            np.random.seed(seeds[0] % (1 << 32))
            torch.manual_seed(seeds[0])
            torch.cuda.manual_seed_all(seeds[0])
            with self._request_seeded_sampler.request_seeds(seeds):
                batch_result = super().infer({"batch": unseeded_observations})
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.set_rng_state(torch_rng_state)
            torch.cuda.set_rng_state_all(cuda_rng_states)

        if not isinstance(batch_result, dict):
            raise RuntimeError(f"LingBot batch returned {type(batch_result).__name__}, expected a mapping")
        results = [dict() for _ in observations]
        for key, values in batch_result.items():
            try:
                value_count = len(values)
            except TypeError as exc:
                raise RuntimeError(f"LingBot batch field {key!r} is not indexable") from exc
            if value_count != len(observations):
                raise RuntimeError(
                    f"LingBot batch field {key!r} has {value_count} values for {len(observations)} requests"
                )
            for index, value in enumerate(values):
                results[index][key] = value
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--robot-config-root", type=pathlib.Path, required=True)
    parser.add_argument("--data-contract", type=pathlib.Path, required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--use-length", type=int, default=5)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--use-bf16", type=str2bool, default=True)
    parser.add_argument("--use-compile", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--torch-threads", type=int, default=LINGBOT_TORCH_THREADS)
    parser.add_argument("--torch-interop-threads", type=int, default=LINGBOT_TORCH_INTEROP_THREADS)
    parser.add_argument("--dynamic-batching", type=str2bool, default=False)
    parser.add_argument("--static-batching", type=str2bool, default=False)
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--lane-count", type=int, default=8)
    parser.add_argument("--deterministic", type=str2bool, default=False)
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=0.0,
        help="Optional request collection window; zero greedily drains only requests already queued",
    )
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be at least 1")
    if args.torch_interop_threads < 1:
        parser.error("--torch-interop-threads must be at least 1")
    if args.max_batch < 1 or args.max_batch & (args.max_batch - 1):
        parser.error("--max-batch must be a positive power of two")
    if args.dynamic_batching and args.static_batching:
        parser.error("--dynamic-batching and --static-batching are mutually exclusive")
    if args.static_batching and (args.lane_count < args.max_batch or args.lane_count % args.max_batch):
        parser.error("--lane-count must be a positive multiple of --max-batch for static batching")
    if args.batch_wait_ms < 0:
        parser.error("--batch-wait-ms must be non-negative")

    # This process primarily performs GPU inference.  Leaving PyTorch at its
    # host-wide default (48 threads on the validator) makes one server consume
    # roughly 11 logical CPUs and starves the MuJoCo clients once every GPU has
    # its own server.  Configure both pools before model construction, compile,
    # or warm-up can initialize them.
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    print(
        f"[lingbot] CPU thread pools: intraop={args.torch_threads} interop={args.torch_interop_threads}",
        flush=True,
    )

    if args.deterministic:
        # CUBLAS_WORKSPACE_CONFIG is also set by run_eval before this process
        # starts; setdefault keeps direct invocations reproducible before the
        # first CUDA context is initialized.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print(
            "[lingbot] strict deterministic algorithms enabled; cuDNN benchmark and TF32 disabled",
            flush=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed % (1 << 32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    policy = EvaluatorLingbotServer(
        args.model_path,
        robot_norm_path=args.norm_stats,
        use_length=args.use_length,
        chunk_ret=True,
        use_bf16=args.use_bf16,
        use_fp32=not args.use_bf16,
        use_compile=args.use_compile,
        robot_config_root=args.robot_config_root,
        data_contract=args.data_contract,
    )
    if (args.dynamic_batching or args.static_batching) and disable_lingbot_batch1_vision_cache(policy):
        # LingBot's optional vision cache is populated with batch-1 split sizes
        # during the first inference and then reused verbatim.  A later batch-2
        # request consequently tries to split four images with two cached
        # lengths.  Dynamic batches must derive those tiny grid tensors from
        # the current shape instead.
        print("[lingbot] disabled batch-1-only precomputed vision grid metadata", flush=True)

    # torch.compile is lazy, so a listening socket alone is not a sufficient
    # readiness signal for this backend. Compile one real-shaped request before
    # opening the port; otherwise the first request blocks the asyncio server
    # while the remaining clients hit websockets' 10-second handshake timeout.
    # Restore every RNG after the synthetic request so warm-up cannot perturb the
    # stochastic action stream used by scored episodes.
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    print("[lingbot] cold-start warm-up: running one contract-shaped inference before serving", flush=True)
    try:
        warmup_duration = warm_up_lingbot_libero_policy(
            policy,
            zeros=np.zeros,
            synchronize=torch.cuda.synchronize,
            image_size=policy.data_config.img_size,
            state_dimension=8,
            minimum_action_steps=args.use_length,
        )
        rng_self_test_duration = verify_lingbot_request_seed_determinism_nonblocking(
            policy,
            on_failure=lambda error: print(
                f"[lingbot] ERROR: {error}; continuing policy server startup",
                file=sys.stderr,
                flush=True,
            ),
            zeros=np.zeros,
            synchronize=torch.cuda.synchronize,
            actions_equal=(
                (lambda left, right: np.array_equal(np.asarray(left), np.asarray(right)))
                if args.deterministic
                else lingbot_actions_are_equivalent
            ),
            seed_field=POLICY_RNG_FIELD,
            image_size=policy.data_config.img_size,
            state_dimension=8,
            minimum_action_steps=args.use_length,
            batch_size=args.max_batch if args.static_batching else 1,
        )
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(torch_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
    print(f"[lingbot] cold-start warm-up complete in {warmup_duration:.1f}s; accepting clients", flush=True)
    if rng_self_test_duration is not None:
        replay_condition = (
            "same-seed replay was bitwise identical"
            if args.deterministic
            else f"same-seed replay stayed within atol={LINGBOT_ACTION_REPLAY_ATOL:g}"
        )
        print(
            f"[lingbot] request-level RNG self-test complete in {rng_self_test_duration:.1f}s; "
            f"{replay_condition} and different seeds diverged",
            flush=True,
        )
    if args.static_batching:
        from lingbot_batch_server import LingbotStaticBatchServer

        print(
            f"[lingbot] static batching enabled: batch_size={args.max_batch} "
            f"lane_count={args.lane_count} deterministic={args.deterministic}",
            flush=True,
        )
        server = LingbotStaticBatchServer(
            policy,
            host="0.0.0.0",
            port=args.port,
            batch_size=args.max_batch,
            lane_count=args.lane_count,
        )
    elif args.dynamic_batching:
        from lingbot_batch_server import LingbotDynamicBatchServer

        print(
            f"[lingbot] dynamic batching enabled: max_batch={args.max_batch} batch_wait_ms={args.batch_wait_ms:g}",
            flush=True,
        )
        server = LingbotDynamicBatchServer(
            policy,
            host="0.0.0.0",
            port=args.port,
            max_batch=args.max_batch,
            batch_wait_ms=args.batch_wait_ms,
        )
    else:
        server = WebsocketPolicyServer(policy, host="0.0.0.0", port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
