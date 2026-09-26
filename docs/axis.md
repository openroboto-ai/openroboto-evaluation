# AXIS v1.0 evaluation protocol

## Current benchmark

`axis_v1.0` is defined by [axis_v1.0.yaml](../configs/benchmarks/axis_v1.0.yaml).
The YAML pins [axis_v1.0.json](../configs/benchmarks/axis_v1.0.json) and its
[30 task snapshots](../configs/benchmarks/axis_v1.0-tasks/). Task IDs, instructions,
initial scenes and success predicates are frozen together.

For a new release, select tasks in the YAML's `tasks` list; its length is the task count. The JSON
stores the frozen definitions and protocol, and the task directory contains each
scene, initial state and success checker. Future 40- or 50-task releases use their
own versioned YAML, JSON and snapshots; published versions are immutable.

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
| Scene randomization | Disabled |
| Renderer | OSMesa |

The evaluator invokes each frozen task's success checker; it does not infer success
from a model response. The current fixed scenes permit task-specific optimization.
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
system OSMesa libraries and verifies the frozen base task bundle.

```bash
uv run python libero_eval/run_eval.py \
  --model /path/to/checkpoint --commit-id local \
  --backbone pi0.5 --benchmark axis_v1.0 \
  --num-trials 20 --gpus 0 --workers-per-gpu 1 \
  --output-dir eval_runs/axis-v1-run
```

Remote repositories require their exact 40-character commit. For a smoke test,
add `--task-ids 501 --num-trials 1`; a subset score is not the full benchmark score.
Use `OPENPI_DIR` and `AXIS_RUNTIME_DIR` to point to separately installed runtimes.

To check the released definitions without a GPU or model:

```bash
uv run python tools/verify_axis_release.py \
  --manifest configs/benchmarks/axis_v1.0.yaml
```

The run stores model identity, definition hashes, actual inference settings,
per-task successes and trial counts in `summary.json`. Keep this result alongside
the exact model revision when comparing runs.

## Project-hosted baseline

The reference checkpoint is available at
[openroboto-ai/pi05-axis-baseline](https://huggingface.co/openroboto-ai/pi05-axis-baseline).
Its verified inference revision is `55f8b28ed021f7ee0bef02cde114a7b5dcae9d5c`.
Use that repository and revision with the evaluation command above. The weights
and normalization files are unchanged; internal training-path records are excluded.
The model card pins evaluator commit `2f69d117517f8b388d2d01a94964df63f1b6620e`,
which includes the replay exporter below.

The included 74.67% result is historical `axis_v0.2` performance on training-task
scenes, not a newly measured AXIS v1.0 result. Hosting a copy does not change an
existing competition's configured baseline repository or revision.

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
