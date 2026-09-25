# RoboDojo reproduction

The `robodojo` benchmark uses the upstream Isaac Sim client and XPolicyLab
Pi 0.5 / Pi 0 adapters, isolated under `third_party/RoboDojo`.

## Install

```bash
bash setup_robodojo.sh
bash setup_robodojo.sh --with-pi0  # Optional Pi 0 adapter
```

The installer pins RoboDojo to `9226f48ea694b3f53db12d4922e8b1199f8d0891`
and the checkpoint/assets dataset to `1a3c4c334aef294c31d7a0190d8d6dff68df78e0`.

## Pi 0.5 protocol

```bash
uv run libero_eval/run_eval.py \
  --benchmark robodojo --model RoboDojo-Benchmark/RoboDojo \
  --model-repo-type dataset \
  --model-subdir 'ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-{seed}' \
  --commit-id 1a3c4c334aef294c31d7a0190d8d6dff68df78e0 \
  --model-architectures pi0.5 --eval-seeds 0,1,2 \
  --gpus 0,1,2,3,4,5,6,7
```

`{seed}` selects one of three training-seed checkpoints; only their directories
are downloaded. The default protocol reports 42 tasks across three seeds.
Each of the 12 generalization tasks runs 25 standard-layout and 25 random-layout
episodes; the remaining 30 tasks run 50 episodes each. This launches 162
simulator runs. Each GPU runs one policy server and one Isaac Sim client at a
time. The server restarts for each task/seed to preserve upstream RNG scheduling.

The frozen Pi-05 reference values are success rate 6.91% and score 11.41.
Dimension scores for Generalization, Precision, Long-Horizon, Memory and Open
are 8.17, 5.50, 14.67, 4.56 and 1.67. The summary uses upstream macro-averaging
and records `evaluation_protocol.official_result`. Reference values are not a
claim that a local run has reproduced them.

## Smoke test and custom runs

```bash
uv run libero_eval/run_eval.py \
  --benchmark robodojo --model RoboDojo-Benchmark/RoboDojo \
  --model-repo-type dataset \
  --model-subdir 'ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0' \
  --commit-id 1a3c4c334aef294c31d7a0190d8d6dff68df78e0 \
  --model-architectures pi0.5 --suites memory --tasks cover_blocks \
  --eval-seeds 0 --num-trials 1 --gpus 0
```

Add `--dry-run` to inspect task selection and commands without Isaac Sim or GPU
locks. For Pi 0, use `ckpt/RoboDojo/Pi_0/RoboDojo-sim-arx_x5-joint-{seed}`.
Select dimensions with `--suites generalization,memory,precision,long-horizon,open`,
tasks with `--tasks`, and layout seeds with `--eval-seeds`. Generalization tasks
are automatically paired with their `_random` variants. Reduced task/seed sets
or explicit trial overrides are marked as development runs.

Seeds affect object poses, clutter, lighting and textures deterministically.
Changes to task definitions or sampling ranges belong in the pinned upstream
YAML configuration; such runs are not directly comparable to its leaderboard.
