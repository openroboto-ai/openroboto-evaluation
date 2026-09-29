"""The numerical runtime contract for native AXIS JAX inference (stdlib only)."""

from collections.abc import MutableMapping


AXIS_JAX_RUNTIME = "axis-jax-deterministic-no-disk-cache-v1"
AXIS_JAX_XLA_FLAGS = (
    "--xla_gpu_deterministic_ops=true --xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_autotune_level=0"
)


def configure_axis_jax_environment(environment: MutableMapping[str, str]) -> None:
    """Apply before importing JAX; ignore inherited executables and tuning choices.

    JAX 0.5.3 enables a per-fusion autotune cache alongside its executable
    cache. Old entries can select different kernels despite deterministic XLA
    flags: our RTX 4090 replay changed 547/600 to 552/600 with a fresh cache.
    Disable both disk cache layers and live tuning. In-process JIT reuse stays
    enabled; only fresh server startup pays compilation cost.
    """
    environment.update({
        "XLA_FLAGS": AXIS_JAX_XLA_FLAGS,
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
    })
    environment.pop("JAX_COMPILATION_CACHE_DIR", None)


def axis_jax_runtime_metadata() -> dict:
    return {
        "policy": AXIS_JAX_RUNTIME,
        "persistent_compilation_cache": False,
        "persistent_autotune_cache": False,
        "autotune_level": 0,
        "xla_flags": AXIS_JAX_XLA_FLAGS,
    }
