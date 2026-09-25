# LingBot-VLA 2.0 with RoboTwin 2.0

This path checks model loading, observation mapping, action decoding, simulation
scheduling and aggregation against the upstream checkpoint protocol.

## Pinned components

| Component | Revision |
|---|---|
| LingBot-VLA 2.0 | `951475ae1b1d87553e7dc47c97b53a3d695c0d13` |
| RoboTwin | `13c3c47ff4312dd62484bcd51be034af55c062d1` |
| Scene assets | `9dc9299c163db059931898a9f0852098a61155a1` |
| Qwen3-VL-4B-Instruct processor/config | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| `robbyant/lingbot-vla-v2-6b-robotwin` | `0451855729ec904f970600e0aec8b84661423afe` |

Use the pinned RoboTwin `script/` layout rather than substituting a newer
XPolicyLab layout. The [upstream README](https://github.com/robbyant/lingbot-vla-v2#robotwin-20)
and checkpoint card report 50 tasks with 100 episodes per task: 93.52% success
for clean and 92.80% for randomized. These README/model-card figures are the
reference target, not a RoboTwin table in the paper.

## Install and run

```bash
bash setup_robotwin.sh --with-assets --with-checkpoint
```

Separate environments are created in `third_party/lingbot-vla-v2/.venv` and
`third_party/RoboTwin/.venv`. Without the flags, only environments are prepared.
The checkpoint is approximately 25.5 GB. Downloads prefer `hfd.sh` with 10
connections and six concurrent files, falling back to HF `snapshot_download`;
both pin the revision. Set `ROBOTWIN_PYTHON=/absolute/path/to/python3.10` to reuse
an interpreter, and `ROBOTWIN_LD_LIBRARY_PATH=/absolute/path/to/conda/lib` if
external dynamic libraries are required and not detected automatically.

```bash
uv run libero_eval/run_eval.py \
  --model hf_models/robbyant__lingbot-vla-v2-6b-robotwin@0451855729ec \
  --commit-id local --backbone lingbot-vla-v2 --benchmark robotwin \
  --robotwin-task-config demo_clean --num-trials 100 \
  --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 1 --save-videos 0
```

Use `demo_randomized` for the randomized protocol. Add `--task-ids 0 --dry-run`
to inspect selection without starting the model or simulator. Resume with
`--resume --output-dir /path/to/previous/run`. Only a complete 50 x 100 run under
the required protocol qualifies for `evaluation_protocol.official_result`.

Keep `--workers-per-gpu 1` on 24 GB RTX 4090 GPUs because SAPIEN/curobo and camera
buffers can exhaust memory with more clients. Larger-memory hardware may use
explicitly tested higher concurrency. Clients share a serial WebSocket policy
server; CPU simulation may overlap GPU inference. Concurrency is recorded, and
scheduling can change flow-matching noise consumption and individual trajectories.

## Reproduction limits

The model card does not disclose the topology used for its reported scores.
At upstream commit `36a9bab235fff53cec13ae4ffd5e7b22e79d0de8`, the launcher
defaulted to eight GPUs with three independent servers each (24 RNG streams).
Its `script/eval_polict_client_openpi.py` is absent from public RoboTwin history.
The public `eval_policy_client_lingbotvla.py` appeared at
`f0f53dee6580a1b23b04646a36e4ff82bdfaf21b`; policy-server code is identical across
the referenced revisions. A complete run supports statistical comparison, not
bit-exact reproduction of an unpublished client.

The summary records local topology, source limitations, protocol completeness
and exact success-count agreement. Its two-proportion z-test treats the two
5,000-episode samples as independent binomial samples; `p >= 0.05` means no
significant difference was detected at that level. This is approximate because
fixed-seed trajectories are not strictly IID. Retain raw counts and percentage
differences rather than relying on the boolean result alone.

The public client uses `instruction_type: unseen` from `deploy_policy.yml`.
`--robotwin-instruction-type seen` is a diagnostic override and sets
`official_result` to false.

Direct HF resolution is also supported:

```bash
uv run libero_eval/run_eval.py \
  --model robbyant/lingbot-vla-v2-6b-robotwin \
  --commit-id 0451855729ec904f970600e0aec8b84661423afe \
  --model-subdir checkpoints/global_step_50000/hf_ckpt \
  --backbone lingbot-vla-v2 --benchmark robotwin --num-trials 100
```

## LIBERO compatibility

RoboTwin uses two arms, three cameras (`camera_top`, `camera_wrist_left`,
`camera_wrist_right`), 14 arm dimensions and two gripper dimensions. This
evaluator's LIBERO contract uses Franka, two cameras (`camera_top`, `camera_wrist`)
and seven action dimensions (six EEF plus one gripper). Changing `--benchmark`
does not adapt these contracts; incompatible RoboTwin weights are rejected
before GPU startup.

Use a LIBERO-trained checkpoint with evaluator-managed normalization:

```bash
uv run libero_eval/run_eval.py \
  --model your-account/lingbot-v2-libero-checkpoint --commit-id PINNED_REVISION \
  --backbone lingbot-vla-v2 --benchmark libero_pro \
  --lingbot-norm-stats /path/to/evaluator/norm_stats.json
```

Joint-target checkpoints such as ManiGuard do not implement LIBERO's relative
EEF contract. See the [upstream Custom Data Guide](https://github.com/robbyant/lingbot-vla-v2/blob/main/lingbotvla/data/vla_data/README.md)
for training data, robot configuration and normalization requirements.
