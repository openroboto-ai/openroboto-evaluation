# LIBERO initial-state sampling

LIBERO and LIBERO-Pro have finite official initial-state lists. The evaluator also
supports reproducible initial states sampled from the environment's original
object-position and orientation distributions. This feature is separate from AXIS
scene randomization; it does not randomize the fixed AXIS v1.0 scenes.

## Direct evaluation

```bash
uv run python libero_eval/run_eval.py \
  --model /path/to/libero-checkpoint --commit-id local \
  --benchmark libero_pro --num-trials 10 --init-seed 12345
```

For each task, the first `min((num_trials + 1) // 2, official_count)` trials use
official initial states. The remaining trials use sampled initial states. The
number `12345` above is a local reproduction example, not a competition seed.

The task-specific seed is
`(seed * 1000003 + crc32(f"{suite}/{task_name}")) % 2**32`.
The implementation is in `libero_eval/init_mix.py`; generation uses
`libero_eval/gen_init_states.py`. Identical versions, input seeds and tasks produce
the same initial states, independent of generation order.

`--init-states-root` replaces the full initial-state source and cannot be combined
with `--init-seed`. Each generated suite stores a manifest with seeds and shapes;
per-trial results record the source used.

## Queue integration

The worker treats the backend's `seed` as an opaque uint32 and passes it to the
evaluator. Seed derivation and publication belong to the backend. A missing or
invalid seed causes a warning and a fallback to official initial states rather
than inventing an unauditable seed. `--no-init-randomization` disables this feature.

Results and score payloads include `init_seed`. Initial states are cached under
the configured cache directory, with separate identities for seed and trial count.
See `tests/test_init_seed.py` for trial allocation, seed derivation and payload tests.
