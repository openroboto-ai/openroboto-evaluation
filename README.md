# OpenRoboto Evaluation

Evaluation harness for AXIS, LIBERO, LIBERO-Pro, LIBERO-plus, RoboTwin and RoboDojo.
Choose a model runtime with `--backbone` and a benchmark with `--benchmark`.

[AXIS protocol](docs/axis.md) · [Queue worker](benchmark_worker/README.md)

## Install

```bash
git clone https://github.com/openroboto-ai/openroboto-evaluation
cd openroboto-evaluation
bash setup.sh
bash setup_axis.sh  # additionally required for AXIS; requires libosmesa6
```

The scheduler uses the root uv environment. Model and simulator dependencies are
isolated under `third_party/`. Installation scripts pin their upstream revisions.
Optional runtimes have separate installers: `setup_lingbot.sh`, `setup_robotwin.sh`
and `setup_robodojo.sh`. Hugging Face downloads use `hfd.sh` when available.

## AXIS v1.0

```bash
uv run python libero_eval/run_eval.py \
  --model your-account/pi05-axis-checkpoint \
  --commit-id 0123456789abcdef0123456789abcdef01234567 \
  --backbone pi0.5 --benchmark axis_v1.0 \
  --num-trials 20 --gpus 0 --workers-per-gpu 1
```

Replace the example model and commit with your exact model revision. A local
checkpoint is also accepted with `--model /path/to/checkpoint --commit-id local`.

The released benchmark contains 30 tasks, 20 trials per task, and a 120-control-step
limit. It uses native 9D joint targets, continuous gripper decoding, and standard
Pi0.5 state-token inputs. The checkpoint must include standard weights and
`assets/axis-v0.1-task501-runtime-v1/norm_stats.json`. Custom training metadata is
not required and cannot override inference. Full details and frozen definitions
are in [the AXIS guide](docs/axis.md).
The asset directory name belongs to the checkpoint format and is independent of the benchmark version.

AXIS v1.0 uses fixed base scenes; scene randomization is disabled. Repeated trials
are not new randomized scenes. Its score measures performance on this published
finite task set, not performance on unseen tasks.

## Other benchmarks

| Benchmark | Runtime / scope |
|---|---|
| `libero` | Four LIBERO suites |
| `libero_pro` | LIBERO-Pro perturbation suites |
| `libero_pro_custom_1` | Worker scoring profile with selected swap suites weighted twice |
| `libero_plus` | LIBERO-plus variants |
| `robotwin` | RoboTwin with the matching LingBot policy and robot contract |
| `robodojo` | RoboDojo with the official policy adapter |

```bash
uv run python libero_eval/run_eval.py \
  --model /path/to/pi05-libero-checkpoint --commit-id local \
  --benchmark libero --num-trials 10 --gpus 0
```

Supported model runtimes include OpenPI Pi0.5 (Orbax or safetensors), OpenVLA-OFT,
and LingBot-VLA 2.0. A checkpoint must match the benchmark's observation, action
and normalization contract; changing a benchmark flag cannot adapt a different robot.
LingBot LIBERO uses evaluator-owned configuration and normalization under
`configs/lingbot_vla_v2/`.

See [LIBERO initial-state sampling](docs/init_state_randomization.md),
[RoboTwin](docs/robotwin.md), and [RoboDojo](docs/robodojo.md).

## Results and validation

Each run writes `summary.json`, per-task results, logs, and optional recordings to
its output directory. The summary records the model revision, benchmark identity,
protocol, and success counts. Invalid checkpoints are rejected before GPU startup;
worker infrastructure failures are retried rather than submitted as model scores.

```bash
uv sync --locked
uv run pytest -q
uv run python tools/verify_axis_release.py \
  --manifest configs/benchmarks/axis_v1.0.yaml
```

The repository contains evaluator code, task definitions, and generic tests.
Internal training trajectories, experiment archives, checkpoints, and machine-specific
service files are not included.
