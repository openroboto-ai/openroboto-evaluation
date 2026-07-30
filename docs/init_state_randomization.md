# Init-State Seed Randomization: Mechanism and Validation Report

## Motivation

The official LIBERO / LIBERO-Pro evaluations freeze each task's initial states in `.pruned_init` files (50 per task; LIBERO-Pro's come from the HF dataset `zhouxueyang/LIBERO-Pro`), making the evaluation set fully deterministic. In a subnet setting, a miner can farm those 50 fixed initial layouts (e.g. generate training data from the evaluation init states), stripping the score of any meaning as a measure of generalization.

## Which dimensions can be randomized

A survey of the LIBERO-PRO source confirmed: **initial object position + rotation is the only "infinite" randomization dimension**.

| Dimension | Nature |
|---|---|
| Initial position + rotation | **Continuous / infinite**: every env reset resamples with `np.random.uniform` inside the BDDL region bounds (`base_region_sampler.py`); the official files simply freeze 50 such samples |
| Object substitution (object) | Finite: 1–6 candidates per class (`ood_object.yaml`, bounded by the asset pool) |
| Spatial-relation swap (swap) | Finite candidate list (`ood_spatial_relation.yaml`) |
| Language rewrite (lan) | Finite: 3 per task (`ood_language.yaml`) |
| Task rewrite (task) | Finite: ~2 per task (`ood_task.yaml`) |
| Environment substitution (env) | Hard-coded to a single target and unstable; not enabled by the validator |

The finite dimensions are already exhausted by the validator's 16 LIBERO-Pro perturbation suites; the incremental anti-overfitting lever is to resample initial positions/rotations with a seed the miner cannot know at training time. The resampling is **same-distribution** as the official inits (identical BDDL sampling bounds), and the deviation magnitude is on the same order as the natural spread within the official 50 states.

## Mechanism and usage

```bash
# 1. Generate init states for a custom seed (client venv, LIBERO-Pro 16 suites)
PYTHONPATH=third_party/LIBERO-PRO \
LIBERO_CONFIG_PATH=~/.cache/libero_pro/config \
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
third_party/openpi/examples/libero/.venv/bin/python \
    libero_eval/gen_init_states.py --seed <SEED> \
    --output-root ~/.cache/libero_pro_custom/init_files_seed<SEED>

# For the base LIBERO suites, switch package and config:
#   PYTHONPATH=third_party/openpi/third_party/libero (do not set LIBERO_CONFIG_PATH)
#   --suites libero_spatial,libero_object,libero_goal,libero_10

# 2. Point the evaluation at the custom init directory
python libero_eval/run_eval.py ... --init-states-root ~/.cache/libero_pro_custom/init_files_seed<SEED>
```

- Each task's seed is derived stably from the base seed (`crc32(suite/task)`), independent of generation order — suites can be generated in parallel, subsets can be regenerated individually, and the output is byte-for-byte reproducible (two runs with the same seed are bitwise identical).
- Every suite directory carries a `manifest.json` (base_seed, per-task derived seed and shape) for auditing; the evaluation's `summary.json` and per-task result JSON record `init_states_root`.
- Generation is fast: 16 suites × 10 tasks × 50 states takes about 10 minutes with 4 parallel processes.

## Validation experiment (2026-07-12)

Models (both PyTorch checkpoints fine-tuned on LIBERO data; evaluation seed=7, 10 trials/task):
- **pi05** = `Fisher-Wang/pi05-libero-pytorch@ab107fbe`
- **pi0** = `Fisher-Wang/pi0-libero-pytorch@23d60043`

Benchmark: LIBERO-Pro 16 perturbation suites (160 tasks, 1600 episodes/run);
control: base LIBERO 4 suites (400 episodes/run). Custom init seed=10007.

### Result 1: resampling introduces no evaluation bias

| run | official | seeded (10007) | delta | p-value |
|---|---|---|---|---|
| pi05 × LIBERO-Pro | 58.9% (943/1600) | 58.1% (930/1600) | +0.8% | 0.64 (within noise) |
| pi0 × LIBERO-Pro | 46.8% (748/1600) | 46.7% (747/1600) | +0.1% | 0.97 (within noise) |
| pi05 × base LIBERO | 96.8% (387/400) | 98.5% (394/400) | −1.7% | 0.10 (within noise) |

Neither model farms the fixed inits, and their scores are unchanged under resampling — matching the expectation that same-distribution resampling is unbiased for honest models. The base-suite control shows that even 95%+ scores do not depend on specific init layouts.

### Result 2: model separability fully preserved

- Ordering unchanged: pi05 > pi0, score gap official +12.2% / seeded +11.4%.
- Task-level success-rate correlation (official vs seeded, 160 tasks):
  pi05 Pearson 0.967 / Spearman 0.894; pi0 Pearson 0.978 / Spearman 0.940.
- Task-level model-gap correlation: Pearson 0.833 / Spearman 0.700.
- (The low task-level correlation in the base control is a ceiling effect: nearly all 40 tasks score 90–100%, so the variance is too small for correlation to be meaningful.)

### Conclusion

Resampling init states under a new seed **loses none of the evaluation's consistency or separability**, while defeating "memorize the initial states" overfitting: honest models score the same, and a model that memorized the fixed layouts falls back to its true generalization level on the resampled half of the evaluation. The scheme is production-ready.

Experiment data: `eval_runs/exp_m{1,2}_pro_{official,seeded}`,
`eval_runs/exp_m1_base_{official,seeded}`;
comparison script: `tools/compare_init_randomization.py`.

## Production integration (implemented; worker default-on)

For every task, half the trials use official inits and half use seed-resampled inits (episode-level 50/50 mix). **The seed is taken directly from the task's queue entry `seed` field** (uint32, one per miner per round), so a miner can reproduce and verify end to end — which is exactly why it was chosen:

1. `seed = int.from_bytes(sha256(f"{block_hash}:{round_num}:{drand_random}")[-4:], "big")`
   is derived by the public protocol (`protocol/seed.py` and `docs/SEED_GENERATION.md` in the subnet repository; drand is independently verifiable at https://api.drand.sh). The block hash comes from the miner's own on-chain commitment and the drand value is fetched at enqueue time, so **a miner cannot know its seed before committing its weights** — there is no "fine-tune on the seed first" overfitting window; after enqueue the seed is publicly queryable and verifiable.
2. `task_seed = (seed * 1000003 + crc32(f"{suite}/{task_name}")) % 2**32`
   (`libero_eval/init_mix.py` `derive_task_seed`; shared test vectors in `tests/test_init_seed.py`).
3. `gen_init_states.py` resamples with `np.random.seed(task_seed)` — byte-for-byte reproducible.
4. Mixing rule: the first `min((num_trials+1)//2, official_available)` trials use official inits, the rest use seeded ones (`init_mix.py` `mix_counts`).

Different miners in the same round get different seeded halves; the experiment above shows seed changes are unbiased and preserve separability, while the official half is identical for all miners, anchoring comparability.

- **`benchmark_worker/worker.py`**: enabled by default. `select_init_seed` reads the queue entry's `seed` field (a missing/invalid value logs a warning and falls back to pure-official evaluation — the worker never invents a local seed, since a privately chosen seed cannot be proven to the miner); the evaluation command automatically appends `--init-seed`. Use `--no-init-randomization` to reproduce the old behavior for debugging.
- **`run_eval.py --init-seed <int>`**: automatically (idempotently, cached by (seed, num_inits)) generates resampled init states for the requested suites (before the policy servers start; tasks are round-robined across GPU worker slots for parallel generation, default 4 generator processes per GPU). Only the states actually needed for the seeded half of the 50/50 mix are generated (50 trials → 25 states; generation is prefix-stable: generating more under the same seed only appends). Production seeds differ per miner per round, so the cache rarely gets reused; a single directory is only a few MB — clear the cache root manually if disk is tight. Mutually exclusive with `--init-states-root` (full replacement, for experiments).
- **`eval_task.py --init-states-mix`**: the first half of trials uses official inits (layouts identical to the first episodes of an unmixed run, preserving comparability; odd trial counts give the official side one extra), the second half uses seeded inits; each episode's `init_source` is recorded in the result JSON.
- **Seed reporting:** the score payload carries an `init_seed` field (= the queue entry seed), which the backend stores in the result for query/confirmation; `summary.json` and the worker state record it as well. Anyone can reproduce all evaluation init states for audit via the derivation chain above with `gen_init_states.py --seed`.

Unit tests: `tests/test_init_seed.py` (trial allocation, `derive_task_seed` shared test vectors, `select_init_seed` selection and fallback, published payload fields).
