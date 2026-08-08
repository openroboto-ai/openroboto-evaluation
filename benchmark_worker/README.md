# Benchmark Worker — Evaluation Orchestration Layer

The isolation layer between the backend Benchmark API and this repository's evaluation pipeline: **the worker main process only does HTTP communication and subprocess management — it runs no evaluation logic**; all evaluation (GPU, policy servers, MuJoCo) happens inside `libero_eval/run_eval.py` subprocesses. The two layers evolve independently; the interface is CLI arguments + `summary.json`.

```
backend (evaluation queue / scoring API, HTTP + X-API-Key)
   ▲  ▲  ▲
   │  └── POST /api/v1/benchmark/task/{id}/score   ← admin key, score submission
   │  └── POST /api/v1/benchmark/task/{id}/progress← admin key, stage/progress
   └───── GET  /api/v1/benchmark/queue             ← public key, periodic polling
   │
benchmark_worker/worker.py (resident main process, this repo's uv environment)
   ├── local persistent queue benchmark_worker_state.json (dedup across restarts)
   ├── model download: huggingface_hub, revision pinned to the task's hf_commit
   │   (tasks with a missing/invalid hf_commit are rejected and reported — unanchored
   │    submissions are never evaluated)
   └── dispatch one by one ──► libero_eval/run_eval.py subprocess (produces summary.json)
```

Interface definitions: see [`docs/api_reference_en.md`](https://github.com/openroboto-ai/openroboto-subnet/blob/main/docs/api_reference_en.md) §3 in the subnet repository.

## Running

From the repository root:

```bash
BACKEND_PUBLIC_API_KEY=<public_key from control.json> \
BACKEND_ADMIN_API_KEY=<admin_key from the backend's backend.yaml> \
    uv run benchmark_worker/worker.py --backend-url http://localhost:8001 \
    --benchmark libero_pro_custom_1 --num-trials 50
```

Authentication is split by least privilege: `GET /api/v1/benchmark/queue` and task details use the `public_key` (`control.json` → `public_key`); the score POST uses the `admin_key` (`backend.yaml` → `admin_key`). Both values are sent via `X-API-Key` and are required at startup. The legacy `--api-key` / `BACKEND_API_KEY` still lets one key do both reads and writes temporarily, but new deployments should use the two separate arguments or environment variables.

While executing a task, the worker reports three stages: `downloading → prechecking → evaluating`. When `evaluating` starts it reports `0/N`, then reports again each time all tasks of a suite complete; `detail` carries `suites_done`, `suites_total`, `last_completed_suite`, `episodes_done`, and `episodes_total`. An unavailable progress endpoint only logs a warning — it never changes evaluation or scoring results. `worker_id` defaults to the hostname and can be set explicitly with `--worker-id` or `BENCHMARK_WORKER_ID`.

Debugging: `--once` (poll once, process everything, exit) combined with `--num-trials 2 --task-ids 0 --gpus 0` gives a single-GPU smoke test; `--allow-local-model` lets `hf_repo_id` be a local checkpoint path directly (testing only).

## Task state machine (benchmark_worker_state.json)

```
pending → running → done_pending_submit → submitted (terminal)
                                        ↘ abandoned (permanently rejected by backend, terminal)
```

- A `task_id` already seen in the queue is never re-evaluated (across polls and restarts);
- If the process is killed: tasks that were mid-evaluation re-run automatically on next start; finished-but-unconfirmed ones are re-submitted;
- Failed submissions back off persistently at 1/2/4/8/16/30 minutes (across polls/restarts); only indeterminate outcomes — timeouts, disconnects, and 5xx — trigger a task-detail recheck, which strictly verifies task identity, terminal state, total score, and the full environment summary before treating the submission as successful; legacy backends may omit `benchmark` in task details, but when the remote does return the field it must match;
- Re-evaluating the same task resets the old result's submission count, errors, and backoff, so a fresh result never inherits historical failure state;
- Definitive 400/404/413/422 responses skip the recheck and go to `abandoned`; 401/403/429 keep the result and wait for configuration fixes or rate-limit recovery.
- The exact JSON body of every POST is logged to the console and `logs/benchmark_worker-YYYY-MM-DD.log`; API keys exist only in HTTP headers and are never written to that log.

## Score mapping (summary.json → score payload)

| API field | Source |
|----------|------|
| `benchmark` | The worker's `--benchmark`; the scoring-profile identity travels with every score submission |
| `env_scores[]` | Normally one entry per summary suite; Custom 1 folds 16 entries into 6 compatibility records per the rules below |
| `total_score` | For Custom 1, the equal-weight mean of the six compatibility records; for other profiles, a local scoring preview value |
| `per_task_scores[]` | Successful entries from summary `tasks` (`libero_spatial_03` format), kept in full in local artifacts/ledger; omitted from the API submission (optional field — avoids the backend writing rows one by one and tripping Cloudflare 524) |
| `duration_sec` | End-to-end wall-clock time for the task (including model download) |
| `success` | Whether a usable score was produced; individual tasks that still fail after retries don't fail the whole run — the loss is recorded in `error` / `env_scores[].error` |

The benchmark is selected entirely by the `--benchmark` startup argument; the queue's `env_list` is a database-compatibility field the worker does not read. `libero_pro` runs and reports the full 16 Pro suites; `libero_pro_custom_1` runs exactly the same 16 suites but, under that profile only, folds the results into 6 compatibility records: each of the four base suites takes the equal-weight mean of its object/swap/lan/task variants — LIBERO-10 is reported as `libero_10` per the backend's current convention (libero_90 is not involved) — plus `libero_object_swap` and `libero_spatial_swap` reported separately. The two swap suites therefore both contribute to the base averages and appear as standalone records.
`libero_plus` forces 10,030 tasks × 1 trial, disables init-state resampling, and rejects task subsets; the four suites are weighted by the official task counts 2402/2518/2591/2519, equivalent to a full-episode micro-average. Unofficial or incomplete LIBERO-Plus summaries are never reported. After upgrading to `libero_plus_official_v1`, cached results under the old protocol in the local ledger auto-invalidate and wait for re-evaluation rather than being re-submitted.
When the same task's local-ledger benchmark doesn't match, the old result is never re-sent.

## Main arguments

| Argument | Default | Description |
|------|------|------|
| `--backend-url` | `$BACKEND_URL` / `http://localhost:8001` | Backend address |
| `--public-api-key` | `$BACKEND_PUBLIC_API_KEY` | Read-only public key (`control.json` → `public_key`) |
| `--admin-api-key` | `$BACKEND_ADMIN_API_KEY` | Write admin key (`backend.yaml` → `admin_key`) |
| `--api-key` | `$BACKEND_API_KEY` | Legacy compatibility: one key for both reads and writes |
| `--queue-path` | `/api/v1/benchmark/queue` | Upstream public task-queue path |
| `--worker-id` | hostname | Worker identity in progress reports; `$BENCHMARK_WORKER_ID` also works |
| `--benchmark` | required | `libero` / `libero_pro` / `libero_pro_custom_1` / `libero_plus` |
| `--poll-interval` | 60 | Queue polling interval (seconds) |
| `--num-trials` | required | Trials per task; LIBERO-Plus must be 1, other LIBERO official protocol is 50 |
| `--gpus` | 0-7 | Passed through to run_eval.py |
| `--init-workers-per-gpu` | 4 | Concurrent init-state generator processes per GPU; split by task, independent of the evaluation's `--workers-per-gpu` |
| `--no-init-randomization` | off | Disable mixed init-state randomization (default on: uses the queue entry's own seed (publicly verifiable, miner-reproducible); per task, half the trials use official inits and half use resampled ones; the seed is returned with the score payload; see docs/init_state_randomization.md) |
| `--eval-timeout` | 28800 | Single-evaluation timeout (seconds) |
| `--state-file` | `benchmark_worker_state.json` | Local state ledger |
| `--output-root` | `eval_runs/` | Evaluation artifacts (logs/videos/summary) |
| `--download-dir` | `hf_models/` | HF model download cache |
