# AXIS V2.0 evaluation protocol

The current release uses [combined randomization](axis_v2_randomization.md)
in native MuJoCo. The V1.0 base-scene bundle remains available as source material
for task import, fixed controls and training replay exports.

## Current benchmark

`axis_v2.0` contains 30 tasks and their frozen randomized instances in
[the release bundle](../configs/benchmarks/axis_v2.0.tar.zst).
Its [receipt](../configs/benchmarks/axis_v2.0-release.json) pins the archive
and manifest hashes. The named CLI verifies and unpacks it into `.cache/axis/releases/axis_v2.0`.
Task IDs, instructions, randomization components and success predicates are frozen together.

| Parameter | Value |
|---|---|
| Trials | 20 per task, 600 total |
| Maximum controls | 120 per trial |
| Control period | 0.2 seconds |
| Replanning | Every 10 controls |
| Action | 9D absolute joint targets, including two fingers and seven arm joints |
| Gripper decoding | Continuous |
| Policy seed | 20260907 |
| Pi0.5 state tokens | Enabled for every submission |
| Scene randomization | All supported components combined |
| Environment seed | Queue-provided; explicit seed for independent comparisons |
| Renderer | OSMesa |

The evaluator invokes each frozen task's success checker; it does not infer success
from a model response. Frozen public instances still permit task-specific optimization.
Publishing the evaluator does not make this protocol an unseen-task test.

## Checkpoint contract

Use standard OpenPI `params/` or `model.safetensors` weights and
`assets/axis-v0.1-task501-runtime-v1/norm_stats.json`.
The asset directory is a checkpoint-format identifier, independent of the benchmark version.
Normalization uses the OpenPI `norm_stats.state` and `norm_stats.actions` structure;
both AXIS entries are 9D and contain `mean`, `std`, `q01`, and `q99`.
Do not relocate AXIS statistics to the LIBERO asset path.

`axis_vla_metadata.json` is not a required submission file. Training sidecars never
select inference inputs; optional training provenance is recorded separately from
actual `discrete_state_input` in the result. Weight and normalization checks run
before policy startup.

## Run

Install with `bash setup.sh` followed by `bash setup_axis.sh`. The latter requires
system OSMesa libraries and verifies the current randomized bundle.

```bash
uv run python libero_eval/run_eval.py \
  --model /path/to/checkpoint --commit-id local \
  --backbone pi0.5 --benchmark axis_v2.0 --axis-randomization-seed 20260928 \
  --num-trials 20 --gpus 0 --workers-per-gpu 1 \
  --output-dir eval_runs/axis-v2-run
```

Remote repositories require their exact 40-character commit. For a smoke test,
add `--task-ids 501 --num-trials 1`; a subset score is not the full benchmark score.
Use `OPENPI_DIR` and `AXIS_RUNTIME_DIR` to point to separately installed runtimes.

To check the released definitions without a GPU or model:

```bash
uv run python -c 'from libero_eval.axis_release import prepare_release; print(prepare_release())'
```

The run stores model identity, definition hashes, actual inference settings,
per-task successes and trial counts in `summary.json`. Keep this result alongside
the exact model revision when comparing runs.

## JAX numerical reproducibility

Native AXIS JAX servers use `axis-jax-deterministic-no-disk-cache-v1`, recorded in
the managed evaluator summary's `numerical_runtime` and the server handshake's
`axis_numerical_runtime`. Both the evaluator launcher and direct
`serve_axis_openpi.py` entry point enforce this configuration before importing JAX:

- Disable persistent JAX executable caches and XLA autotune caches, ignoring
  inherited cache paths and XLA flags.
- Disable live autotuning with `--xla_gpu_autotune_level=0` and require
  deterministic GPU operations.
- Retain in-process JIT reuse. Each fresh policy process must compile again;
  default kernels may also be slower than tuned kernels.

This addresses a measured RTX 4090 discrepancy: identical model weights, seed
and scenes scored 547/600 with an older autotune cache and 552/600 with a fresh
cache. Deterministic-operation flags alone did not isolate those stored choices.
Scores from the old cache-dependent runtime must be reevaluated together before
comparison with this runtime; changing runtime does not preserve an old score.
The analysis tool rejects comparisons mixing recorded numerical runtime policies,
including a new policy with an unrecorded legacy policy.

This is not a cross-hardware guarantee. Keep the checkpoint, evaluator, OpenPI,
JAX/jaxlib, CUDA libraries, simulator, CPU/rendering environment and GPU model
fixed when testing exact replay. RTX 4090 versus RTX 5090 equivalence requires
separate measurement; the same seed or container does not establish it. Compare
per-episode outcomes, terminal steps and checker values, not only the total score.
PyTorch checkpoints and external policy servers are outside this JAX contract;
their evaluator summary records `numerical_runtime: null`.

See [XLA determinism](https://openxla.org/xla/determinism) and
[cuBLAS reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)
for the distinction between deterministic execution and numerical portability.

## Project-hosted baseline

The reference checkpoint is available at
[openroboto-ai/pi05-axis-baseline](https://huggingface.co/openroboto-ai/pi05-axis-baseline).
Its verified inference revision is `55f8b28ed021f7ee0bef02cde114a7b5dcae9d5c`.
The weights and normalization files are unchanged.
The model card pins evaluator commit `2f69d117517f8b388d2d01a94964df63f1b6620e`.
Its historical result is not a newly measured V2.0 randomized score.
Publishing this evaluator does not change a competition's configured baseline.

## Render training replays

[export_axis_vla_dataset.py](../tools/export_axis_vla_dataset.py) renders expert
joint-target trajectories through the same `AxisEnvironment`, frozen scenes,
initial states and success checkers used by evaluation. It uses OSMesa and the
scene's `camera0`; it does not copy images from the source dataset or convert
an existing camera view into another view. No model checkpoint or policy GPU
is needed. Install the AXIS runtime as described above.

The input is a local, unpacked AXIS Franka Zarr store with these arrays:

| Array | Contract |
|---|---|
| `data/state` | `[T, 9]` joint positions |
| `data/action` | `[T, 9]` absolute joint-position targets |
| `meta/episode_ends` | Increasing cumulative frame counts, ending at `T` |

The joint order must match `runtime.observation_joint_order` in the manifest:
two finger joints followed by the seven arm joints. Use trajectories for the
specified task and base scene. The default source rate is 30 Hz; stride 6 feeds
targets to the benchmark's 5 Hz controller. If your source rate differs, set
both `--source-frequency-hz` and `--stride`; their ratio must match the frozen
control period. Other dataset layouts, robots or action conventions require an
explicit conversion before using this tool.

From the repository root:

```bash
MUJOCO_GL=osmesa "${AXIS_RUNTIME_DIR:-third_party/axis-runtime}/.venv/bin/python" \
  tools/export_axis_vla_dataset.py \
  --dataset /path/to/task_501_isaac_state_train_100.zarr \
  --task-id 501 \
  --manifest configs/benchmarks/axis_v1.0.yaml \
  --output datasets/axis_v1.0/task501-replays.npz
```

The default manifest is `axis_v1.0.yaml`. Task definitions come from its pinned
snapshots. Scene assets are downloaded on first use into `.cache/axis`; retain
that cache for later offline runs. Use `--cache-root` to choose another location.

By default the tool checks every source episode and exports only successful
replays. `--episodes 0,3` restricts the candidates to those zero-based indices.
Each selected successful episode is replayed again for capture; the script
fails if none pass or if capture no longer succeeds. It resets to the frozen
initial scene and records the image and simulated joint state **before** each
target, stopping at success or the end of the source episode. Training replays
can be longer than the 120-control policy evaluation limit.

The `.npz` output contains `images` (`uint8 [T,256,256,3]` for v1.0), `states`
and `actions` (`float32 [T,9]`), `episode_ends`, and `metadata_json`.
Metadata records the task instruction, source episode indices, sampling rate,
camera, renderer, manifest/task hashes and an artifact checksum. Load it with
`axis_vla.load_artifact`; `AxisReplayDataset` supplies the instruction as the
prompt and builds action chunks without crossing episode boundaries.
[AxisInputs](../libero_eval/axis_vla.py) maps the rendered image to `base_0_rgb`
and masks the absent wrist views; [pi05_axis_joint](../libero_eval/axis_openpi_config.py)
defines the shared OpenPI transforms and quantile normalization. Checkpoints
still need their matching normalization statistics.

These are training demonstrations on the published task scenes. The artifact's
`eligible_for_scoring=false` marks replay data, not a restriction on models
trained with it. Measure a resulting checkpoint with the evaluation command
above; replay success counts are not model scores or held-out generalization
results. The tool exports the replay artifact, not a trained checkpoint.

## Generated versions

The worker discovers complete `axis_v*.yaml` bundles. A generated version must keep
its YAML, pinned manifest and adjacent task snapshots together. Existing versions
cannot be overwritten with different definitions. Later task selection is generated
from an operator-supplied pinned selector and runtime pool; neither unpublished
selection results nor internal training material are bundled here.

See [the worker guide](../benchmark_worker/README.md) for queue routing and optional
version preparation. Generation alone does not activate or publish a competition.
