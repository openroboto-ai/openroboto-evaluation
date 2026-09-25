"""Dynamic OpenPI config for the native AXIS 9D joint contract.

This lives in the validator so the pinned upstream OpenPI checkout can remain
unmodified.  Import it only from an OpenPI environment.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
from typing import Any

from flax import nnx
from openpi.models import model as openpi_model
from openpi.models import pi0_config
from openpi.training import config as openpi_config
from openpi.training import optimizer
from openpi.training import weight_loaders
from openpi import transforms

from axis_vla import AxisInputs, AxisOutputs
from axis_training_scope import TRAINING_SCOPES, action_expert_parameter
from axis_training_ema import validate_ema_decay


CONFIG_NAME = "pi05_axis_joint"
ASSET_ID = "axis-v0.1-task501-runtime-v1"
DATASET_PREFIX = "axis-runtime-replay:"


class ActionExpertTrainConfig(openpi_config.TrainConfig):
    """Restrict optimizer updates while preserving the backbone's original dtype.

    Upstream initialization casts `freeze_filter` matches to bfloat16. Keep that
    filter empty and restrict `trainable_filter` instead, so initialization does
    not round the frozen visual/language weights before the first update.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.freeze_filter, nnx.Nothing):
            raise ValueError("action-expert training requires an empty dtype-conversion freeze_filter")

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, action_expert_parameter)


@dataclasses.dataclass(frozen=True)
class AxisDataConfigFactory(openpi_config.DataConfigFactory):
    repo_id: str = DATASET_PREFIX
    asset_id: str = ASSET_ID
    supplied_norm_stats: dict[str, Any] | None = None
    gripper_mode: str = "continuous"

    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: openpi_model.BaseModelConfig,
    ) -> openpi_config.DataConfig:
        del assets_dirs
        return openpi_config.DataConfig(
            repo_id=self.repo_id,
            asset_id=self.asset_id,
            norm_stats=self.supplied_norm_stats,
            data_transforms=transforms.Group(inputs=[AxisInputs()], outputs=[AxisOutputs(self.gripper_mode)]),
            model_transforms=openpi_config.ModelTransformFactory()(model_config),
            use_quantile_norm=True,
        )


def make_config(
    *,
    dataset_path: pathlib.Path | None = None,
    norm_stats: dict[str, Any] | None = None,
    init_params: str = "gs://openpi-assets/checkpoints/pi05_libero/params",
    checkpoint_base_dir: pathlib.Path | str = "./checkpoints",
    exp_name: str = "axis_task501_runtime_vla",
    num_train_steps: int = 1_000,
    batch_size: int = 8,
    num_workers: int = 0,
    fsdp_devices: int = 1,
    seed: int = 42,
    overwrite: bool = False,
    gripper_mode: str = "continuous",
    discrete_state_input: bool = False,
    peak_lr: float = 5e-5,
    decay_lr: float = 5e-6,
    training_scope: str = "full",
    ema_decay: float | None = None,
) -> openpi_config.TrainConfig:
    validate_ema_decay(ema_decay)
    if not math.isfinite(peak_lr) or not math.isfinite(decay_lr) or not 0 < decay_lr <= peak_lr:
        raise ValueError("learning rates must be finite and satisfy 0 < decay_lr <= peak_lr")
    if type(discrete_state_input) is not bool:
        raise ValueError("discrete_state_input must be a boolean")
    if training_scope not in TRAINING_SCOPES:
        raise ValueError(f"training_scope must be one of {TRAINING_SCOPES}")
    repo_id = DATASET_PREFIX + (str(pathlib.Path(dataset_path).resolve()) if dataset_path is not None else "unused")
    config_class = ActionExpertTrainConfig if training_scope == "action-expert" else openpi_config.TrainConfig
    return config_class(
        name=CONFIG_NAME,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=discrete_state_input),
        data=AxisDataConfigFactory(repo_id=repo_id, supplied_norm_stats=norm_stats, gripper_mode=gripper_mode),
        weight_loader=weight_loaders.CheckpointWeightLoader(init_params),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=min(200, max(1, num_train_steps // 5)),
            peak_lr=peak_lr,
            decay_steps=max(num_train_steps, 1),
            decay_lr=decay_lr,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=ema_decay,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        num_workers=num_workers,
        save_interval=num_train_steps + 1,
        keep_period=None,
        fsdp_devices=fsdp_devices,
        wandb_enabled=False,
        checkpoint_base_dir=str(pathlib.Path(checkpoint_base_dir).resolve()),
        exp_name=exp_name,
        seed=seed,
        overwrite=overwrite,
        policy_metadata={
            "benchmark": "axis_v1.0",
            "action_contract": "absolute 9D joint-position targets",
            "axis_gripper_mode": gripper_mode,
            "axis_discrete_state_input": discrete_state_input,
            "purpose": "benchmark-validation-vla",
            "eligible_for_scoring": False,
        },
    )
