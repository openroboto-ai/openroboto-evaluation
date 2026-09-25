"""Initialize OpenPI with sharded checkpoint inputs to avoid a replicated-weight peak.

Adapted from OpenPI scripts/train.py (15a9616a). The initialization and optimizer
semantics are preserved; checkpoint inputs use the output parameter sharding so
buffer donation can reuse them on GPUs with limited memory.
"""

from __future__ import annotations


def select_parameter_sharding(parameters, sharding):
    """Project a full parameter sharding tree onto a partially loaded checkpoint."""
    if isinstance(parameters, dict):
        return {key: select_parameter_sharding(value, sharding[key]) for key, value in parameters.items()}
    return sharding


def init_train_state(config, init_rng, mesh, *, resume):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.shared import nnx_utils
    from openpi.training import optimizer, sharding, utils as training_utils
    from scripts import train

    tx = optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(shape, mesh, log=True)
    if resume:
        return shape, state_sharding
    partial_params = train._load_weights_and_validate(config.weight_loader, shape.params.to_pure_dict())
    input_sharding = select_parameter_sharding(partial_params, state_sharding.params.to_pure_dict())
    partial_params = jax.device_put(partial_params, input_sharding)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=(replicated, input_sharding),
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return state, state_sharding
