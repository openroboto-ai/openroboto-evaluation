# Benchmark queue worker

The worker polls an authenticated backend, downloads each exact model revision,
checks compatibility, runs the evaluator, and submits complete task results.
`run_eval.py` can also be used independently without a queue or backend credentials.

## Start a worker

Set `BACKEND_PUBLIC_API_KEY` and `BACKEND_ADMIN_API_KEY` in the process environment
or a private service environment file. Do not put real keys in shell examples or
commit populated environment files.

```bash
uv run python benchmark_worker/worker.py \
  --backend-url https://backend.example.invalid \
  --axis-only --num-trials 20 --gpus 0 --workers-per-gpu 1 \
  --server-impl upstream \
  --state-file eval_runs/worker-state.json --output-root eval_runs/worker
```

`--axis-only` filters the queue to AXIS without replacing each task's benchmark
version. The queue's `base_model` selects the model runtime. Use a separate state,
output and generated-version directory for each backend. The worker verifies its
committed source revision for provenance; start it from a clean Git checkout.

Invalid checkpoints fail before GPU startup. GPU/runtime infrastructure failures
are retried without charging partial scores to the submitted model. Progress,
per-task results and model identities are preserved for resumption and audit.

## Optional AXIS version preparation

Provide `WORKER_KEY`, `--axis-selector-root /path/to/pinned-selector` and
`--axis-runtime-pool /path/to/frozen-pool.json` to enable preparation from the
backend's `GET /api/v1/benchmark/rotation` response. Both source directories must
be supplied together. The pinned runtime pool includes its task snapshots and
must resolve scene assets; cloning a selector alone is not an environment setup.

A non-null rotation response specifies `benchmark`, `previous_benchmark`, `seed`,
`seed_block`, and `opens_at`. The worker verifies and installs complete consecutive
versions atomically, retains older versions, and never rewrites an existing task
set. Unsupported versions are skipped before score/progress submission. It does
not publish competitions or choose a new baseline on the backend.

Use `--axis-benchmark-dir /path/to/versions` to select storage explicitly. Do not
combine automatic preparation with a benchmark override. The backend owns timing,
seed publication and activation. Generic service templates are in
[deploy/systemd](../deploy/systemd/README.md).

For the evaluator's current task and normalization contract, see
[AXIS](../docs/axis.md). LIBERO workers can use the queue-provided initial-state seed
as described in [initial-state sampling](../docs/init_state_randomization.md).

Run `uv run python benchmark_worker/worker.py --help` for all scheduling, download,
GPU concurrency, retry and output options.
