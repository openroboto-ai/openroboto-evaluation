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

## Generated versions

The worker discovers complete `axis_v*.yaml` bundles. A generated version must keep
its YAML, pinned manifest and adjacent task snapshots together. Existing versions
cannot be overwritten with different definitions. Later task selection is generated
from an operator-supplied pinned selector and runtime pool; neither unpublished
selection results nor internal training material are bundled here.

See [the worker guide](../benchmark_worker/README.md) for queue routing and optional
version preparation. Generation alone does not activate or publish a competition.
