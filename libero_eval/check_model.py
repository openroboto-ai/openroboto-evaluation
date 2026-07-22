"""Pre-evaluation legality checks for supported policy checkpoints.

Validates that a model directory is a well-formed openpi checkpoint *before*
any GPU time is spent on it, and reports every violation with a human-readable
reason. Mirrors what `openpi.policies.policy_config.create_trained_policy`
will actually try to load:

  - PyTorch checkpoint: `model.safetensors` (takes precedence when present)
  - JAX checkpoint: `params/` (orbax OCDBT PyTree)
  - both: `assets/<asset_id>/norm_stats.json` with the stats the config needs
    (pi05 uses quantile normalization, so q01/q99 are mandatory)

Checks are pure-stdlib (no jax/torch): safetensors headers are parsed by hand,
orbax structure is read from `params/_METADATA`. This runs in the lightweight
validator env, not the openpi server venv.

Standalone usage (resolves local path / HF repo id / HF URL like run_eval.py;
HF references must be pinned with --commit-id, full 40-hex sha):
    uv run libero_eval/check_model.py --model <path-or-hf-repo> [--commit-id <sha>] \
        [--model-architectures pi0,pi0.5] [--config pi05_libero] [--json]

Exit code: 0 = valid, 1 = rejected (reasons printed).
"""

import argparse
import ast
import dataclasses
import json
import math
import pathlib
import struct
import sys
import typing

MODEL_FAMILIES = ("openpi", "openvla_oft")


# ----------------------------------------------------------------------------
# What a valid checkpoint looks like, per openpi training config
# ----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ConfigSpec:
    """Format requirements implied by an openpi TrainConfig (see openpi training/config.py)."""

    pi05: bool  # pi05 arch (time_mlp_*) vs pi0 (state_proj/action_time_mlp_*)
    action_dim: int  # Pi0Config.action_dim
    action_horizon: int  # Pi0Config.action_horizon (informational; no param shape depends on it)
    expert_width: int  # action expert hidden width (gemma_300m -> 1024)
    asset_id: str  # norm stats live in assets/<asset_id>/norm_stats.json
    use_quantile_norm: bool  # pi05/fast configs normalize with q01/q99, pi0 with mean/std
    state_dim: int  # dim of the state the eval client sends (LIBERO: 8)
    norm_dims: dict  # canonical norm_stats array lengths, e.g. {"state": 8, "actions": 7}


def _libero_spec(*, pi05: bool, action_horizon: int, use_quantile_norm: bool) -> ConfigSpec:
    return ConfigSpec(
        pi05=pi05,
        action_horizon=action_horizon,
        use_quantile_norm=use_quantile_norm,
        action_dim=32,
        expert_width=1024,
        asset_id="physical-intelligence/libero",
        state_dim=8,
        norm_dims={"state": 8, "actions": 7},
    )


CONFIG_SPECS = {
    "pi05_libero": _libero_spec(pi05=True, action_horizon=10, use_quantile_norm=True),
    "pi0_libero": _libero_spec(pi05=False, action_horizon=50, use_quantile_norm=False),
    "pi0_libero_low_mem_finetune": _libero_spec(pi05=False, action_horizon=50, use_quantile_norm=False),
}

ARCHITECTURE_CONFIGS = {
    "pi0": "pi0_libero",
    "pi0.5": "pi05_libero",
}
_ARCHITECTURE_ALIASES = {
    "pi0": "pi0",
    "π0": "pi0",
    "pi0.5": "pi0.5",
    "pi05": "pi0.5",
    "π0.5": "pi0.5",
}

# Total parameter count of PaliGemma(2B+400M vision) + gemma_300m expert + projections.
# Reference checkpoints: 3.35e9 (JAX), 3.62e9 (PyTorch, stores the tied lm_head twice).
_PARAM_COUNT_RANGE = (2.5e9, 4.5e9)

_SAFETENSORS_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


@dataclasses.dataclass
class CheckResult:
    checkpoint_dir: str
    config: str
    checkpoint_type: str | None = None  # "pytorch" | "jax" | None (not a checkpoint)
    errors: list = dataclasses.field(default_factory=list)
    warnings: list = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclasses.dataclass(frozen=True)
class ModelSelection:
    """The validation result plus the config selected for inference."""

    result: CheckResult
    architecture: str | None
    config: str | None


def detect_model_family(ckpt_dir: pathlib.Path | str) -> str:
    """Identify a supported checkpoint family from stable on-disk markers."""
    root = pathlib.Path(ckpt_dir)
    config = None
    try:
        config = json.loads((root / "config.json").read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        pass
    if (
        isinstance(config, dict)
        and config.get("model_type") == "openvla"
        and (root / "model.safetensors.index.json").is_file()
    ):
        return "openvla_oft"
    if (root / "model.safetensors").is_file() or (root / "params").is_dir():
        return "openpi"
    raise ValueError(
        "unsupported checkpoint format: expected an openpi model (model.safetensors or params/) "
        "or OpenVLA-OFT model (model_type=openvla plus model.safetensors.index.json)"
    )


def check_openvla_oft_model(ckpt_dir: pathlib.Path | str) -> CheckResult:
    """Validate the sharded OpenVLA-OFT package used by the official OFT+ model."""
    root = pathlib.Path(ckpt_dir)
    res = CheckResult(checkpoint_dir=str(root), config="openvla_oft_libero", checkpoint_type="openvla_oft")
    if not root.is_dir():
        res.errors.append(f"checkpoint path is not a directory: {root}")
        return res

    config = _load_json(root / "config.json", res, "OpenVLA config")
    if config is not None:
        if config.get("model_type") != "openvla":
            res.errors.append(f"config.json model_type must be 'openvla', got {config.get('model_type')!r}")
        architectures = config.get("architectures")
        if not isinstance(architectures, list) or "OpenVLAForActionPrediction" not in architectures:
            res.errors.append("config.json architectures must include 'OpenVLAForActionPrediction'")

    index = _load_json(root / "model.safetensors.index.json", res, "sharded safetensors index")
    if index is not None:
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            res.errors.append("model.safetensors.index.json must contain a non-empty weight_map")
        else:
            shards = sorted(set(weight_map.values()))
            if not all(isinstance(name, str) and name.endswith(".safetensors") for name in shards):
                res.errors.append("model.safetensors.index.json weight_map contains invalid shard names")
            for name in shards:
                if not isinstance(name, str):
                    continue
                path = root / name
                if not path.is_file():
                    res.errors.append(f"missing weight shard referenced by index: {name}")
                elif _is_lfs_pointer(path):
                    res.errors.append(f"weight shard {name} is a git-lfs pointer, not the real weights")
                elif path.stat().st_size < 1024 * 1024:
                    res.errors.append(f"weight shard {name} is implausibly small ({path.stat().st_size} bytes)")

    for pattern, label in (
        ("action_head--*_checkpoint.pt", "action head"),
        ("proprio_projector--*_checkpoint.pt", "proprio projector"),
    ):
        matches = list(root.glob(pattern))
        if len(matches) != 1:
            res.errors.append(f"expected exactly one {label} file matching {pattern!r}, found {len(matches)}")
        elif _is_lfs_pointer(matches[0]):
            res.errors.append(f"{matches[0].name} is a git-lfs pointer, not the real weights")

    for name in (
        "dataset_statistics.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "modeling_prismatic.py",
        "configuration_prismatic.py",
        "processing_prismatic.py",
    ):
        if not (root / name).is_file():
            res.errors.append(f"missing required OpenVLA-OFT file: {name}")

    stats = _load_json(root / "dataset_statistics.json", res, "dataset statistics")
    if stats is not None:
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            suite_stats = stats.get(suite)
            if not isinstance(suite_stats, dict):
                res.errors.append(f"dataset_statistics.json is missing suite {suite!r}")
                continue
            for key, dims in (("action", 7), ("proprio", 8)):
                values = suite_stats.get(key)
                if not isinstance(values, dict):
                    res.errors.append(f"dataset statistics {suite}.{key} must be an object")
                    continue
                for stat in ("q01", "q99"):
                    array = values.get(stat)
                    if not _finite_number_list(array) or len(array) != dims:
                        res.errors.append(
                            f"dataset statistics {suite}.{key}.{stat} must contain {dims} finite numbers"
                        )
    return res


def parse_model_architectures(value: str) -> tuple[str, ...]:
    """Parse a comma-separated architecture allow-list, accepting ASCII or π spellings."""
    architectures = []
    for raw in value.split(","):
        token = raw.strip().lower()
        if not token:
            raise ValueError("model architectures must be a non-empty comma-separated list")
        try:
            architecture = _ARCHITECTURE_ALIASES[token]
        except KeyError as e:
            raise ValueError(
                f"unknown model architecture '{raw.strip()}'; choose from pi0,pi0.5 "
                "(comma-separated to accept more than one)"
            ) from e
        if architecture not in architectures:
            architectures.append(architecture)
    return tuple(architectures)


def architecture_for_config(config: str) -> str | None:
    spec = CONFIG_SPECS.get(config)
    if spec is None:
        return None
    return "pi0.5" if spec.pi05 else "pi0"


def check_model_for_architectures(
    ckpt_dir: pathlib.Path | str,
    architectures: tuple[str, ...],
    config: str | None = None,
) -> ModelSelection:
    """Validate a checkpoint against an architecture allow-list and select its inference config.

    An explicit config remains authoritative. Without one, each allowed architecture's
    canonical LIBERO config is tried and the matching config is returned. A well-formed
    checkpoint of a disallowed architecture gets a concise allow-list rejection instead
    of a misleading collection of missing/unexpected-module errors.
    """
    if not architectures:
        raise ValueError("at least one model architecture must be accepted")

    if config is not None:
        architecture = architecture_for_config(config)
        if architecture is not None and architecture not in architectures:
            res = CheckResult(checkpoint_dir=str(ckpt_dir), config=config)
            res.errors.append(
                f"config '{config}' uses architecture '{architecture}', which is not accepted "
                f"(accepted: {','.join(architectures)})"
            )
            return ModelSelection(res, architecture, None)
        return ModelSelection(check_model(ckpt_dir, config), architecture, config)

    allowed_results = []
    for architecture in architectures:
        candidate_config = ARCHITECTURE_CONFIGS[architecture]
        result = check_model(ckpt_dir, candidate_config)
        if result.ok:
            return ModelSelection(result, architecture, candidate_config)
        allowed_results.append((result, architecture, candidate_config))

    # Detect a valid but disallowed architecture so the filtering decision is explicit.
    for architecture, candidate_config in ARCHITECTURE_CONFIGS.items():
        if architecture in architectures:
            continue
        result = check_model(ckpt_dir, candidate_config)
        if result.ok:
            result.errors.append(
                f"checkpoint architecture '{architecture}' is not accepted "
                f"(accepted: {','.join(architectures)})"
            )
            return ModelSelection(result, architecture, None)

    # For malformed checkpoints, report the closest allowed format rather than
    # multiplying nearly identical errors from every candidate architecture.
    result, architecture, candidate_config = min(allowed_results, key=lambda item: len(item[0].errors))
    return ModelSelection(result, architecture, candidate_config)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def check_model(ckpt_dir: pathlib.Path | str, config: str = "pi05_libero") -> CheckResult:
    """Validate an openpi checkpoint directory. Returns all problems found (empty errors = valid)."""
    ckpt_dir = pathlib.Path(ckpt_dir)
    res = CheckResult(checkpoint_dir=str(ckpt_dir), config=config)

    if not ckpt_dir.is_dir():
        res.errors.append(f"checkpoint path is not a directory: {ckpt_dir}")
        return res

    spec = CONFIG_SPECS.get(config)
    if spec is None:
        res.warnings.append(
            f"no format spec for config '{config}' (known: {sorted(CONFIG_SPECS)}); "
            "only generic structure checks were performed"
        )

    has_pytorch = (ckpt_dir / "model.safetensors").is_file()
    has_jax = (ckpt_dir / "params").is_dir()
    if not has_pytorch and not has_jax:
        res.errors.append(
            "not an openpi checkpoint: expected PyTorch weights ('model.safetensors') "
            "or a JAX orbax checkpoint ('params/' directory) at the checkpoint root"
        )
        return res
    if has_pytorch and has_jax:
        res.warnings.append(
            "both 'model.safetensors' and 'params/' exist; openpi loads the PyTorch weights and ignores 'params/'"
        )

    # Mirror openpi: model.safetensors takes precedence over params/.
    if has_pytorch:
        res.checkpoint_type = "pytorch"
        _check_pytorch_weights(ckpt_dir / "model.safetensors", spec, res)
        _check_pytorch_config_json(ckpt_dir / "config.json", spec, res)
    else:
        res.checkpoint_type = "jax"
        _check_jax_params(ckpt_dir / "params", spec, res)

    if spec is not None:
        _check_norm_stats(ckpt_dir, spec, res)
    return res


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def _is_lfs_pointer(path: pathlib.Path) -> bool:
    """Detect an un-downloaded `git lfs` pointer file left in place of the real content."""
    try:
        if path.stat().st_size > 1024:
            return False
        return path.read_bytes().startswith(b"version https://git-lfs")
    except OSError:
        return False


def _load_json(path: pathlib.Path, res: CheckResult, what: str) -> dict | None:
    if _is_lfs_pointer(path):
        res.errors.append(
            f"{what} ({path.name}) is a git-lfs pointer file, not the real content — the upload/download is incomplete"
        )
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        res.errors.append(f"{what} ({path.name}) is not valid JSON: {e}")
        return None
    if not isinstance(data, dict):
        res.errors.append(f"{what} ({path.name}) must be a JSON object, got {type(data).__name__}")
        return None
    return data


def _finite_number_list(v) -> "typing.TypeGuard[list[float]]":
    return isinstance(v, list) and len(v) > 0 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)


def _dir_total_bytes(path: pathlib.Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _expected_proj_shapes(spec: ConfigSpec, *, torch: bool) -> dict:
    """Shapes of the small projection layers, which pin down action_dim and expert width.

    torch nn.Linear weight is [out, in]; JAX nnx.Linear kernel is [in, out].
    """
    w, a = spec.expert_width, spec.action_dim

    def lin(name, din, dout) -> dict:
        if torch:
            return {f"{name}.weight": [dout, din], f"{name}.bias": [dout]}
        return {(name, "kernel"): [din, dout], (name, "bias"): [dout]}

    shapes = {**lin("action_in_proj", a, w), **lin("action_out_proj", w, a)}
    if spec.pi05:
        shapes.update(lin("time_mlp_in", w, w))
        shapes.update(lin("time_mlp_out", w, w))
    else:
        shapes.update(lin("state_proj", a, w))
        shapes.update(lin("action_time_mlp_in", 2 * w, w))
        shapes.update(lin("action_time_mlp_out", w, w))
    return shapes


def _module_mismatch_errors(found: set, expected: set, res: CheckResult, where: str) -> None:
    if missing := sorted(expected - found):
        res.errors.append(f"{where}: missing parameter groups {missing} (expected {sorted(expected)})")
    if extra := sorted(found - expected):
        hint = ""
        if {"state_proj", "action_time_mlp_in"} & set(extra):
            hint = " — these belong to the pi0 architecture; this looks like a pi0 checkpoint, not pi05"
        elif {"time_mlp_in", "time_mlp_out"} & set(extra):
            hint = " — these belong to the pi05 architecture; this looks like a pi05 checkpoint, not pi0"
        res.errors.append(f"{where}: unexpected parameter groups {extra}{hint}")


def _check_param_count(total: int, res: CheckResult, where: str) -> None:
    lo, hi = _PARAM_COUNT_RANGE
    if not lo <= total <= hi:
        res.errors.append(
            f"{where}: total parameter count {total:.2e} is outside the plausible range "
            f"[{lo:.1e}, {hi:.1e}] for a PaliGemma+expert openpi model — wrong or truncated weights"
        )


# ----------------------------------------------------------------------------
# PyTorch checkpoint (model.safetensors)
# ----------------------------------------------------------------------------
def _check_pytorch_weights(st_path: pathlib.Path, spec: ConfigSpec | None, res: CheckResult) -> None:
    if _is_lfs_pointer(st_path):
        res.errors.append(
            "model.safetensors is a git-lfs pointer file, not the real weights — the upload/download is incomplete"
        )
        return
    file_size = st_path.stat().st_size
    if file_size < 16:
        res.errors.append(f"model.safetensors is too small to be a safetensors file ({file_size} bytes)")
        return
    with open(st_path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        if header_len > 100 * 1024 * 1024 or 8 + header_len > file_size:
            res.errors.append(
                f"model.safetensors has a corrupt header (declared header size {header_len} bytes, "
                f"file size {file_size} bytes)"
            )
            return
        try:
            header = json.loads(f.read(header_len))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            res.errors.append(f"model.safetensors header is not valid JSON: {e}")
            return
    if not isinstance(header, dict):
        res.errors.append("model.safetensors header must be a JSON object")
        return

    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        res.errors.append("model.safetensors contains no tensors")
        return

    data_size = file_size - 8 - header_len
    spans, total_params = [], 0
    for name, info in sorted(tensors.items()):
        if not isinstance(info, dict) or not {"dtype", "shape", "data_offsets"} <= info.keys():
            res.errors.append(f"model.safetensors: tensor '{name}' lacks dtype/shape/data_offsets")
            return
        dtype, shape, offsets = info["dtype"], info["shape"], info["data_offsets"]
        if dtype not in _SAFETENSORS_DTYPE_BYTES:
            res.errors.append(f"model.safetensors: tensor '{name}' has unknown dtype '{dtype}'")
            return
        if not (isinstance(shape, list) and all(isinstance(d, int) and d >= 0 for d in shape)):
            res.errors.append(f"model.safetensors: tensor '{name}' has invalid shape {shape}")
            return
        begin, end = offsets
        nbytes = math.prod(shape) * _SAFETENSORS_DTYPE_BYTES[dtype]
        if not (0 <= begin <= end <= data_size) or end - begin != nbytes:
            res.errors.append(
                f"model.safetensors: tensor '{name}' data offsets {offsets} do not match its "
                f"shape/dtype ({nbytes} bytes expected, data section is {data_size} bytes) "
                "— file is corrupt or truncated"
            )
            return
        total_params += math.prod(shape)
        spans.append((begin, end))

    spans.sort()
    pos = 0
    for begin, end in spans:
        if begin != pos:
            res.errors.append(
                "model.safetensors: tensor data does not tile the data section contiguously "
                f"(gap/overlap at byte {pos}) — file is corrupt"
            )
            return
        pos = end
    if pos != data_size:
        res.errors.append(
            f"model.safetensors: data section is {data_size} bytes but tensors only cover {pos} "
            "— file is corrupt or truncated"
        )
        return

    if spec is None:
        return
    # openpi loads with safetensors.torch.load_model(PI0Pytorch(config), path), which
    # requires the key set/shapes to match the architecture built from --config.
    _module_mismatch_errors(
        {name.split(".")[0] for name in tensors},
        {"paligemma_with_expert", *(k.split(".")[0] for k in _expected_proj_shapes(spec, torch=True))},
        res,
        "model.safetensors",
    )
    for name, want in _expected_proj_shapes(spec, torch=True).items():
        if name in tensors and tensors[name]["shape"] != want:
            res.errors.append(
                f"model.safetensors: tensor '{name}' has shape {tensors[name]['shape']}, expected {want} "
                f"(config '{res.config}': action_dim={spec.action_dim}, expert width={spec.expert_width})"
            )
    _check_param_count(total_params, res, "model.safetensors")


def _check_pytorch_config_json(cfg_path: pathlib.Path, spec: ConfigSpec | None, res: CheckResult) -> None:
    """config.json is informational (openpi builds the model from --config, not from it),
    so contradictions are reported as warnings — the weight shapes are the ground truth."""
    if spec is None or not cfg_path.is_file():
        return
    cfg = _load_json(cfg_path, res, "config.json")
    if cfg is None:
        return
    for key, expected in [("action_dim", spec.action_dim), ("action_horizon", spec.action_horizon)]:
        if key in cfg and cfg[key] != expected:
            res.warnings.append(
                f"config.json declares {key}={cfg[key]} but config '{res.config}' uses {expected}; "
                "the checkpoint may have been trained with a different config"
            )


# ----------------------------------------------------------------------------
# JAX checkpoint (params/, orbax OCDBT)
# ----------------------------------------------------------------------------
def _check_jax_params(params_dir: pathlib.Path, spec: ConfigSpec | None, res: CheckResult) -> None:
    meta_path = params_dir / "_METADATA"
    if not meta_path.is_file():
        res.errors.append(
            "params/_METADATA is missing — orbax cannot restore a checkpoint without it "
            "(the checkpoint save was incomplete or files were dropped during upload)"
        )
        return
    metadata = _load_json(meta_path, res, "params/_METADATA")
    if metadata is None:
        return

    tree, shapes = _parse_orbax_tree(metadata)
    if tree is None:
        res.errors.append("params/_METADATA has an unrecognized layout (no readable 'tree_metadata')")
        return
    if not isinstance(tree.get("params"), dict):
        res.errors.append(
            "params/_METADATA tree has no top-level 'params' subtree — openpi's restore_params() "
            f"requires it (found top-level keys: {sorted(tree)})"
        )
        return

    # Array data: OCDBT layout (manifest.ocdbt + d/) or legacy per-array zarr dirs.
    if (params_dir / "manifest.ocdbt").is_file():
        if not any((params_dir / "d").glob("*")):
            res.errors.append(
                "params/d/ contains no array data despite an OCDBT manifest — the checkpoint upload is incomplete"
            )
            return
    elif not any(p.is_dir() for p in params_dir.iterdir()):
        res.errors.append("params/ contains no array data (neither an OCDBT store nor zarr array dirs)")
        return
    if not (params_dir / "commit_success.txt").is_file():
        res.warnings.append(
            "params/commit_success.txt is missing — the orbax save may not have "
            "committed cleanly; loading can still work but verify the source"
        )

    if spec is None:
        return
    _module_mismatch_errors(
        set(tree["params"]),
        {"PaliGemma", *(k[0] for k in _expected_proj_shapes(spec, torch=False))},
        res,
        "params/ (orbax tree)",
    )
    if shapes:
        for key, want in _expected_proj_shapes(spec, torch=False).items():
            got = shapes.get(("params", *key))
            if got is not None and got != want:
                res.errors.append(
                    f"params/: array {'/'.join(('params', *key))} has shape {got}, expected {want} "
                    f"(config '{res.config}': action_dim={spec.action_dim}, expert width={spec.expert_width})"
                )
        total = sum(math.prod(s) for s in shapes.values())
        _check_param_count(total, res, "params/")
        # Cheap truncation guard: even at 1 byte/param the store must be larger than this.
        if (size := _dir_total_bytes(params_dir)) < total:
            res.errors.append(
                f"params/ holds {size / 1e9:.2f} GB for {total:.2e} parameters — too small even for "
                "1 byte/param, the array data is truncated"
            )


def _parse_orbax_tree(metadata: dict) -> tuple[dict | None, dict]:
    """Extract (nested key tree, {key tuple: shape}) from orbax _METADATA.

    Recent orbax writes a flat {"('params', 'a', 'b')": {..., 'value_metadata': {'write_shape': [...]}}}
    mapping under 'tree_metadata'; older versions store the nested tree directly. Shapes may be absent.
    """
    tm = metadata.get("tree_metadata")
    if not isinstance(tm, dict):
        return None, {}

    keys = []
    for k in tm:
        try:
            t = ast.literal_eval(k)
        except (ValueError, SyntaxError):
            keys = None
            break
        if not (isinstance(t, tuple) and all(isinstance(s, str) for s in t)):
            keys = None
            break
        keys.append(t)

    if keys is None:  # nested (legacy) layout: tree_metadata already is the tree
        return tm, {}

    tree: dict = {}
    shapes: dict = {}
    for t, info in zip(keys, tm.values()):
        node = tree
        for part in t[:-1]:
            node = node.setdefault(part, {})
        node[t[-1]] = {}
        if isinstance(info, dict):
            shape = (info.get("value_metadata") or {}).get("write_shape")
            if isinstance(shape, list):
                shapes[t] = shape
    return tree, shapes


# ----------------------------------------------------------------------------
# Normalization stats (assets/<asset_id>/norm_stats.json)
# ----------------------------------------------------------------------------
def _check_norm_stats(ckpt_dir: pathlib.Path, spec: ConfigSpec, res: CheckResult) -> None:
    rel = pathlib.Path("assets") / spec.asset_id / "norm_stats.json"
    path = ckpt_dir / rel
    if not path.is_file():
        res.errors.append(
            f"missing normalization stats: {rel} — openpi loads them from the checkpoint "
            f"(config '{res.config}' uses asset id '{spec.asset_id}')"
        )
        return
    data = _load_json(path, res, str(rel))
    if data is None:
        return
    stats = data.get("norm_stats")
    if not isinstance(stats, dict):
        res.errors.append(f"{rel}: top-level 'norm_stats' key is missing or not an object")
        return

    fields = ["mean", "std"] + (["q01", "q99"] if spec.use_quantile_norm else [])
    for key in ("state", "actions"):
        entry = stats.get(key)
        if not isinstance(entry, dict):
            res.errors.append(
                f"{rel}: norm_stats['{key}'] is missing — inference cannot "
                f"{'normalize inputs' if key == 'state' else 'unnormalize actions'} without it"
            )
            continue
        lengths = set()
        bad = False
        for field in fields:
            v = entry.get(field)
            if not _finite_number_list(v):
                res.errors.append(
                    f"{rel}: norm_stats['{key}']['{field}'] must be a non-empty array of finite "
                    f"numbers{' (quantile normalization needs q01/q99)' if field in ('q01', 'q99') else ''}"
                )
                bad = True
                continue
            lengths.add(len(v))
        if bad:
            continue
        if len(lengths) != 1:
            res.errors.append(f"{rel}: norm_stats['{key}'] fields {fields} have inconsistent lengths")
            continue
        (length,) = lengths
        if key == "state" and length < spec.state_dim:
            res.errors.append(
                f"{rel}: norm_stats['state'] arrays have {length} dims but the eval client sends a "
                f"{spec.state_dim}-dim state — normalization would fail at inference time"
            )
        elif length != spec.norm_dims[key]:
            res.warnings.append(
                f"{rel}: norm_stats['{key}'] arrays have {length} dims (canonical LIBERO stats "
                f"have {spec.norm_dims[key]})"
            )
        if any(x < 0 for x in entry["std"]):
            res.errors.append(f"{rel}: norm_stats['{key}']['std'] contains negative values")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    from paths import VALIDATOR_ROOT  # local import: keep check_model importable standalone

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Local checkpoint dir, HF repo id (user/repo), or HF URL")
    parser.add_argument(
        "--model-family",
        default="auto",
        choices=("auto", *MODEL_FAMILIES),
        help="Checkpoint family (default: infer from checkpoint markers)",
    )
    parser.add_argument(
        "--commit-id",
        default=None,
        help="HF commit hash to check (full 40-hex sha; required for HF references, ignored for local checkpoint dirs)",
    )
    parser.add_argument(
        "--model-architectures",
        default="pi0.5",
        metavar="ARCH[,ARCH...]",
        help="Accepted model architectures (default: pi0.5; e.g. pi0 or pi0,pi0.5; π spellings also work)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Explicit openpi training config name (default: auto-select from the checkpoint architecture)",
    )
    parser.add_argument(
        "--download-dir", default=str(VALIDATOR_ROOT / "hf_models"), help="Where HF models are downloaded"
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON report")
    args = parser.parse_args()

    from run_eval import resolve_model  # lazy: run_eval imports this module at top level

    try:
        architectures = parse_model_architectures(args.model_architectures)
    except ValueError as e:
        parser.error(str(e))

    try:
        ckpt_dir = resolve_model(args.model, pathlib.Path(args.download_dir), commit_id=args.commit_id)
    except Exception as e:  # unresolvable reference is itself a legality failure
        res = CheckResult(
            checkpoint_dir=str(args.model),
            config=args.config or ARCHITECTURE_CONFIGS[architectures[0]],
            errors=[f"cannot resolve model to a checkpoint: {e}"],
        )
        selection = ModelSelection(res, None, None)
    else:
        try:
            family = detect_model_family(ckpt_dir)
        except ValueError as e:
            res = CheckResult(checkpoint_dir=str(ckpt_dir), config=args.config or "auto", errors=[str(e)])
            selection = ModelSelection(res, None, None)
        else:
            if args.model_family != "auto" and args.model_family != family:
                res = CheckResult(
                    checkpoint_dir=str(ckpt_dir),
                    config=args.config or "auto",
                    errors=[f"requested family {args.model_family!r}, detected {family!r}"],
                )
                selection = ModelSelection(res, family, None)
            elif family == "openvla_oft":
                if args.config is not None:
                    res = CheckResult(
                        checkpoint_dir=str(ckpt_dir),
                        config=args.config,
                        errors=["--config is only valid for openpi checkpoints"],
                    )
                else:
                    res = check_openvla_oft_model(ckpt_dir)
                selection = ModelSelection(res, family, res.config if res.ok else None)
            else:
                selection = check_model_for_architectures(ckpt_dir, architectures, args.config)
                res = selection.result

    if args.json:
        print(
            json.dumps(
                {
                    "ok": res.ok,
                    **dataclasses.asdict(res),
                    "architecture": selection.architecture,
                    "selected_config": selection.config,
                },
                indent=2,
            )
        )
    else:
        print(f"checkpoint: {res.checkpoint_dir}")
        print(f"config:     {res.config}   type: {res.checkpoint_type or 'unknown'}")
        if selection.architecture is not None:
            print(f"architecture: {selection.architecture}")
        for w in res.warnings:
            print(f"  warning: {w}")
        for i, e in enumerate(res.errors, 1):
            print(f"  problem {i}: {e}")
        print("RESULT: " + ("VALID" if res.ok else f"REJECTED ({len(res.errors)} problem(s))"))
    sys.exit(0 if res.ok else 1)


if __name__ == "__main__":
    main()
