"""Evaluator-owned LingBot-VLA 2.0 LIBERO runtime contract.

The checkpoint defines the neural-network architecture and weights. Camera,
state/action mapping, and normalization are benchmark inputs, so production
evaluation loads them from this repository rather than trusting files supplied
by a model submitter.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import math
import pathlib
import time
from types import SimpleNamespace
from typing import Callable

import yaml

from lingbot_eval_protocol import validate_policy_seed


VALIDATOR_ROOT = pathlib.Path(__file__).resolve().parents[1]
LINGBOT_CONFIG_ROOT = VALIDATOR_ROOT / "configs" / "lingbot_vla_v2"
DEFAULT_LINGBOT_DATA_CONTRACT = LINGBOT_CONFIG_ROOT / "libero_data.yaml"
DEFAULT_LINGBOT_NORM_STATS = LINGBOT_CONFIG_ROOT / "libero_norm_stats.json"
DEFAULT_LINGBOT_ROBOT_CONFIG_ROOT = LINGBOT_CONFIG_ROOT / "robot_configs"
LINGBOT_TORCH_THREADS = 4
LINGBOT_TORCH_INTEROP_THREADS = 1
# TorchInductor otherwise creates up to 32 compiler workers per process.  One
# policy server is started per GPU, so the default would fan out to 224 workers
# on the seven-GPU validator during cold compilation.
LINGBOT_COMPILE_THREADS = 4

_EXPECTED_JOINTS = {"end.position": 14, "effector.position": 2}
_EXPECTED_CAMERAS = ("camera_top", "camera_wrist")
_EXPECTED_NORM_TYPES = {
    "end.position": "bounds_99_woclip",
    "effector.position": "bounds_99_woclip",
}
_EXPECTED_NORM_DIMS = {
    "action.end.position": 6,
    "action.effector.position": 1,
    "observation.state.end.position": 6,
    "observation.state.effector.position": 2,
}

_LIBERO_WARMUP_PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"
# Across the reproduced four-GPU BF16 runs, same-seed replay drift peaked at
# 0.0127 in action units, while different seeds differed by up to 0.319.
# Keep a measured safety margin without allowing a missing seed to pass.
LINGBOT_ACTION_REPLAY_ATOL = 0.02


class LingbotRequestSeedSelfTestFailure(RuntimeError):
    """The RNG smoke test's action comparison violated its heuristic."""


def copy_readonly_observation_arrays(observation: dict) -> dict:
    """Copy read-only array values before PyTorch creates tensors from them.

    LingBot's msgpack decoder returns NumPy views backed by immutable message
    buffers. ``torch.as_tensor`` and ``torch.from_numpy`` otherwise retain that
    read-only storage and warn that later writes would have undefined behavior.
    This helper uses NumPy's array protocol attributes without importing NumPy,
    keeping the scheduler-side runtime module lightweight.
    """
    copied = dict(observation)
    for feature, value in copied.items():
        flags = getattr(value, "flags", None)
        if flags is not None and not bool(getattr(flags, "writeable", True)):
            copied[feature] = value.copy()
    return copied


def disable_lingbot_batch1_vision_cache(policy) -> bool:
    """Disable fixed-shape vision metadata before LingBot's first dynamic batch.

    The cache flag used at inference lives on the nested vision/expert model,
    not only on the checkpoint-level config.  That model may already be wrapped
    in ``torch.compile``, so update the original module and clear any metadata
    left by an eager probe.
    """
    outer_model = policy.vla.model
    vision_model = outer_model.qwenvl_with_expert
    while hasattr(vision_model, "_orig_mod"):
        vision_model = vision_model._orig_mod

    disabled = False
    for config in (
        getattr(policy, "config", None),
        getattr(policy.vla, "config", None),
        getattr(outer_model, "config", None),
        getattr(vision_model, "config", None),
    ):
        if config is not None and getattr(config, "precompute_grid_thw", False):
            config.precompute_grid_thw = False
            disabled = True

    for field in (
        "pos_embeds",
        "position_embeddings",
        "cu_seqlens",
        "visual_split_sizes",
        "visual_max_seqlen",
    ):
        if getattr(vision_model, field, None) is not None:
            setattr(vision_model, field, None)
            disabled = True
    return disabled


def lingbot_actions_are_equivalent(left: object, right: object) -> bool:
    """Accept bounded BF16 kernel drift while still distinguishing RNG seeds."""
    if hasattr(left, "tolist"):
        left = left.tolist()
    if hasattr(right, "tolist"):
        right = right.tolist()
    left_is_sequence = isinstance(left, (list, tuple))
    right_is_sequence = isinstance(right, (list, tuple))
    if left_is_sequence or right_is_sequence:
        if not left_is_sequence or not right_is_sequence or len(left) != len(right):
            return False
        return all(lingbot_actions_are_equivalent(a, b) for a, b in zip(left, right))
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left_number)
        and math.isfinite(right_number)
        and abs(left_number - right_number) <= LINGBOT_ACTION_REPLAY_ATOL
    )


def lingbot_action_drift_summary(left: object, right: object) -> str:
    """Return compact numeric evidence for a failed same-seed comparison."""
    differences: list[float] = []
    shape_matches = True
    non_finite = 0

    def visit(left_value: object, right_value: object) -> None:
        nonlocal shape_matches, non_finite
        if hasattr(left_value, "tolist"):
            left_value = left_value.tolist()
        if hasattr(right_value, "tolist"):
            right_value = right_value.tolist()
        left_sequence = isinstance(left_value, (list, tuple))
        right_sequence = isinstance(right_value, (list, tuple))
        if left_sequence or right_sequence:
            if not left_sequence or not right_sequence or len(left_value) != len(right_value):
                shape_matches = False
                return
            for left_item, right_item in zip(left_value, right_value):
                visit(left_item, right_item)
            return
        try:
            difference = abs(float(left_value) - float(right_value))
        except (TypeError, ValueError):
            shape_matches = False
            return
        if math.isfinite(difference):
            differences.append(difference)
        else:
            non_finite += 1

    visit(left, right)
    if not differences:
        return f"shape_match={shape_matches}, comparable_values=0, non_finite={non_finite}"
    changed = sum(difference != 0.0 for difference in differences)
    mean_difference = sum(differences) / len(differences)
    return (
        f"shape_match={shape_matches}, max_abs_diff={max(differences):.8g}, "
        f"mean_abs_diff={mean_difference:.8g}, changed={changed}/{len(differences)}, "
        f"non_finite={non_finite}, within_atol_{LINGBOT_ACTION_REPLAY_ATOL:g}="
        f"{non_finite == 0 and max(differences) <= LINGBOT_ACTION_REPLAY_ATOL}"
    )


class LingbotRequestSeededSampler:
    """Supply explicit action noise to LingBot's compiled sampling function.

    Keeping ``torch.randn`` outside the compiled model makes request-level RNG
    independent of compiler-internal random offsets. Unseeded calls (including
    cold-start warm-up) still use torch's default generator, while scored calls
    use a fresh device-local generator derived only from the request seed.
    """

    def __init__(
        self,
        sample_actions: Callable,
        *,
        generator_factory: Callable,
        randn: Callable,
        concatenate: Callable | None = None,
        action_steps: int,
        action_dimension: int,
    ):
        if not callable(sample_actions):
            raise TypeError("LingBot sample_actions must be callable")
        if not isinstance(action_steps, int) or isinstance(action_steps, bool) or action_steps <= 0:
            raise ValueError(f"LingBot action_steps must be a positive integer, got {action_steps!r}")
        if not isinstance(action_dimension, int) or isinstance(action_dimension, bool) or action_dimension <= 0:
            raise ValueError(f"LingBot action_dimension must be a positive integer, got {action_dimension!r}")
        self._sample_actions = sample_actions
        self._generator_factory = generator_factory
        self._randn = randn
        self._concatenate = concatenate
        self._action_steps = action_steps
        self._action_dimension = action_dimension
        self._active_seed: int | tuple[int, ...] | None = None

    @contextlib.contextmanager
    def request_seed(self, seed: int):
        """Activate one seed for exactly one synchronous inference request."""
        seed = validate_policy_seed(seed)
        if self._active_seed is not None:
            raise RuntimeError("LingBot request seed scopes cannot be nested")
        self._active_seed = seed
        try:
            yield
        finally:
            self._active_seed = None

    @contextlib.contextmanager
    def request_seeds(self, seeds):
        """Activate one independent seed per item in a synchronous batch."""
        validated = tuple(validate_policy_seed(seed) for seed in seeds)
        if not validated:
            raise ValueError("LingBot batch request seeds cannot be empty")
        if self._active_seed is not None:
            raise RuntimeError("LingBot request seed scopes cannot be nested")
        self._active_seed = validated
        try:
            yield
        finally:
            self._active_seed = None

    def __call__(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        *,
        image_grid_thw=None,
    ):
        batch_size = int(state.shape[0])
        noise_shape = (batch_size, self._action_steps, self._action_dimension)
        noise_kwargs = {"device": state.device, "dtype": state.dtype}
        if isinstance(self._active_seed, int):
            generator = self._generator_factory(device=state.device)
            generator.manual_seed(self._active_seed)
            noise_kwargs["generator"] = generator
            noise = self._randn(noise_shape, **noise_kwargs)
        elif isinstance(self._active_seed, tuple):
            if len(self._active_seed) != batch_size:
                raise ValueError(f"LingBot batch has {batch_size} samples but {len(self._active_seed)} request seeds")
            per_sample_noise = []
            for seed in self._active_seed:
                generator = self._generator_factory(device=state.device)
                generator.manual_seed(seed)
                per_sample_noise.append(
                    self._randn(
                        (1, self._action_steps, self._action_dimension),
                        device=state.device,
                        dtype=state.dtype,
                        generator=generator,
                    )
                )
            if len(per_sample_noise) == 1:
                noise = per_sample_noise[0]
            else:
                if self._concatenate is None:
                    raise RuntimeError("LingBot batched request seeds require a tensor concatenate function")
                noise = self._concatenate(per_sample_noise, dim=0)
        else:
            noise = self._randn(noise_shape, **noise_kwargs)
        return self._sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            image_grid_thw=image_grid_thw,
        )


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_named_values(values, field: str) -> dict:
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list")
    parsed = {}
    for index, raw in enumerate(values):
        try:
            item = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{field}[{index}] is not a one-entry mapping: {raw!r}") from exc
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"{field}[{index}] is not a one-entry mapping: {raw!r}")
        name, value = next(iter(item.items()))
        if name in parsed:
            raise ValueError(f"{field} contains duplicate entry {name!r}")
        parsed[name] = value
    return parsed


def load_data_contract(path: pathlib.Path | str = DEFAULT_LINGBOT_DATA_CONTRACT) -> SimpleNamespace:
    """Load and fail closed on any drift from the trained LIBERO contract."""
    path = pathlib.Path(path)
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read LingBot LIBERO data contract {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"LingBot LIBERO data contract must be a mapping: {path}")

    joints = _parse_named_values(payload.get("joints"), "joints")
    norm_types = _parse_named_values(payload.get("norm_type"), "norm_type")
    cameras = payload.get("cameras")
    if joints != _EXPECTED_JOINTS:
        raise ValueError(f"LingBot LIBERO joints must be {_EXPECTED_JOINTS!r}, got {joints!r}")
    if cameras != list(_EXPECTED_CAMERAS):
        raise ValueError(f"LingBot LIBERO cameras must be {list(_EXPECTED_CAMERAS)!r}, got {cameras!r}")
    if norm_types != _EXPECTED_NORM_TYPES:
        raise ValueError(f"LingBot LIBERO norm types must be {_EXPECTED_NORM_TYPES!r}, got {norm_types!r}")
    if payload.get("img_size") != 256:
        raise ValueError(f"LingBot LIBERO img_size must be 256, got {payload.get('img_size')!r}")

    # Upstream FeatureInfo expects the argparse-normalized string form rather
    # than mappings produced directly by yaml.safe_load.
    return SimpleNamespace(
        joints=[repr({name: dimension}) for name, dimension in joints.items()],
        cameras=list(cameras),
        norm_type=[repr({name: norm_type}) for name, norm_type in norm_types.items()],
        img_size=256,
    )


def validate_norm_stats(path: pathlib.Path | str = DEFAULT_LINGBOT_NORM_STATS) -> dict:
    """Validate the fixed 7D action / 8D state normalization artifact."""
    path = pathlib.Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read LingBot LIBERO norm stats {path}: {exc}") from exc
    stats = payload.get("norm_stats") if isinstance(payload, dict) else None
    if not isinstance(stats, dict):
        raise ValueError(f"LingBot LIBERO norm stats must contain a norm_stats object: {path}")
    if set(stats) != set(_EXPECTED_NORM_DIMS):
        raise ValueError(
            f"LingBot LIBERO norm stats features must be exactly {sorted(_EXPECTED_NORM_DIMS)}, got {sorted(stats)}"
        )
    for feature, dimension in _EXPECTED_NORM_DIMS.items():
        feature_stats = stats[feature]
        if not isinstance(feature_stats, dict):
            raise ValueError(f"norm stats {feature} must be an object")
        for statistic in ("mean", "std", "q01", "q99", "min", "max"):
            values = feature_stats.get(statistic)
            if (
                not isinstance(values, list)
                or len(values) != dimension
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in values
                )
            ):
                raise ValueError(f"norm stats {feature}.{statistic} must contain {dimension} finite numbers")
    count = payload.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError(f"LingBot LIBERO norm stats count must be a positive integer, got {count!r}")
    return payload


def runtime_contract_metadata(
    data_contract: pathlib.Path | str,
    norm_stats: pathlib.Path | str,
) -> dict[str, str | int]:
    data_contract = pathlib.Path(data_contract)
    norm_stats = pathlib.Path(norm_stats)
    payload = validate_norm_stats(norm_stats)
    load_data_contract(data_contract)
    return {
        "data_contract": str(data_contract),
        "data_contract_sha256": sha256_file(data_contract),
        "norm_stats": str(norm_stats),
        "norm_stats_sha256": sha256_file(norm_stats),
        "norm_stats_count": payload["count"],
    }


def warm_up_lingbot_libero_policy(
    policy,
    *,
    zeros: Callable,
    synchronize: Callable[[], None],
    image_size: int = 256,
    state_dimension: int = 8,
    minimum_action_steps: int = 5,
) -> float:
    """Run one contract-shaped inference before the WebSocket server listens.

    LingBot's ``torch.compile`` is lazy: constructing the compiled callable does
    not compile it. If multiple evaluation clients are started immediately after
    the TCP port opens, the first inference blocks the server event loop while
    the other clients are still handshaking. Keeping this warm-up server-side
    makes readiness mean "the first inference completed", not merely "the model
    weights loaded".

    ``zeros`` and ``synchronize`` are injected so this boundary logic remains
    unit-testable in the validator's lightweight scheduler environment. The
    LingBot server supplies NumPy and CUDA implementations at runtime.
    """
    if image_size <= 0:
        raise ValueError(f"LingBot warm-up image_size must be positive, got {image_size}")
    if state_dimension <= 0:
        raise ValueError(f"LingBot warm-up state_dimension must be positive, got {state_dimension}")
    if minimum_action_steps <= 0:
        raise ValueError(f"LingBot warm-up minimum_action_steps must be positive, got {minimum_action_steps}")

    reset_request = {"reset": True, "robo_name": "libero"}
    observation = {
        "observation.image": zeros((image_size, image_size, 3), dtype="uint8"),
        "observation.wrist_image": zeros((image_size, image_size, 3), dtype="uint8"),
        "observation.state": zeros((state_dimension,), dtype="float32"),
        "task": _LIBERO_WARMUP_PROMPT,
    }

    policy.infer(reset_request)
    started = time.monotonic()
    try:
        result = policy.infer(observation)
        synchronize()
        if not isinstance(result, dict) or "action" not in result:
            raise RuntimeError("LingBot warm-up inference did not return an 'action' field")
        try:
            action_steps = len(result["action"])
        except TypeError as exc:
            raise RuntimeError("LingBot warm-up action is not a step sequence") from exc
        if action_steps < minimum_action_steps:
            raise RuntimeError(
                f"LingBot warm-up returned {action_steps} action steps; expected at least {minimum_action_steps}"
            )
    finally:
        # Do not leak the synthetic rollout's chunk/global-step state into the
        # first scored episode. A warm-up failure exits the server afterwards.
        policy.infer(reset_request)
    return time.monotonic() - started


def verify_lingbot_request_seed_determinism(
    policy,
    *,
    zeros: Callable,
    synchronize: Callable[[], None],
    actions_equal: Callable[[object, object], bool],
    seed_field: str,
    image_size: int = 256,
    state_dimension: int = 8,
    minimum_action_steps: int = 5,
    batch_size: int = 1,
) -> float:
    """Check bounded same-seed replay and distinguish an interleaved seed."""
    if not isinstance(seed_field, str) or not seed_field:
        raise ValueError(f"LingBot request seed field must be a non-empty string, got {seed_field!r}")
    if image_size <= 0 or state_dimension <= 0 or minimum_action_steps <= 0 or batch_size <= 0:
        raise ValueError("LingBot request-seed self-test dimensions must be positive")

    reset_request = {"reset": True, "robo_name": "libero"}
    observation = {
        "observation.image": zeros((image_size, image_size, 3), dtype="uint8"),
        "observation.wrist_image": zeros((image_size, image_size, 3), dtype="uint8"),
        "observation.state": zeros((state_dimension,), dtype="float32"),
        "task": _LIBERO_WARMUP_PROMPT,
    }

    def sample(seed: int):
        policy.infer(reset_request)
        requests = []
        for position in range(batch_size):
            request = dict(observation)
            request[seed_field] = seed + position
            requests.append(request)
        results = [policy.infer(requests[0])] if batch_size == 1 else policy.infer_batch(requests)
        synchronize()
        if not isinstance(results, (list, tuple)) or len(results) != batch_size:
            raise RuntimeError(
                f"LingBot request-seed self-test returned {type(results).__name__}, "
                f"expected {batch_size} result mappings"
            )
        actions = []
        for result in results:
            if not isinstance(result, dict) or "action" not in result:
                raise RuntimeError("LingBot request-seed self-test did not return an 'action' field")
            try:
                action_steps = len(result["action"])
            except TypeError as exc:
                raise RuntimeError("LingBot request-seed self-test action is not a step sequence") from exc
            if action_steps < minimum_action_steps:
                raise RuntimeError(
                    f"LingBot request-seed self-test returned {action_steps} action steps; "
                    f"expected at least {minimum_action_steps}"
                )
            actions.append(result["action"])
        return actions[0] if batch_size == 1 else actions

    started = time.monotonic()
    try:
        first = sample(0x13579BDF2468ACE)
        interleaved = sample(0x2468ACE13579BDF)
        if actions_equal(first, interleaved):
            raise LingbotRequestSeedSelfTestFailure(
                "LingBot request-level RNG self-test failed: different seeds produced equivalent actions"
            )
        repeated = sample(0x13579BDF2468ACE)
        if not actions_equal(first, repeated):
            raise LingbotRequestSeedSelfTestFailure(
                "LingBot request-level RNG self-test failed: same-seed replay drift exceeded tolerance "
                f"({lingbot_action_drift_summary(first, repeated)})"
            )
    finally:
        policy.infer(reset_request)
    return time.monotonic() - started


def verify_lingbot_request_seed_determinism_nonblocking(
    policy,
    *,
    on_failure: Callable[[str], None],
    **kwargs,
) -> float | None:
    """Run the RNG heuristic without turning observed drift into a startup failure.

    Only the two explicit pass/fail judgments are softened. Exceptions raised
    while producing an action, basic response-contract failures, and invalid
    checker configuration still propagate so a genuinely unusable policy
    server cannot advertise itself as healthy.
    """
    try:
        return verify_lingbot_request_seed_determinism(policy, **kwargs)
    except LingbotRequestSeedSelfTestFailure as exc:
        on_failure(str(exc))
        return None
