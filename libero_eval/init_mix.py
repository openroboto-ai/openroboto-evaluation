"""Episode allocation and seed derivation for mixed official/seeded init-state
evaluation.

Pure stdlib and Python-3.8-safe: imported by eval_task.py / gen_init_states.py
(LIBERO client venv) and by the validator's tests (repo venv), so the 50/50
split and the per-task seed formula each have a single definition.

derive_task_seed is the published "translate" function: given a queue entry's
seed (uint32, itself derived from block_hash + round_num + drand — see the
prototype repo's docs/SEED_GENERATION.md and protocol/seed.py), any miner can
recompute the exact per-task seeds the validator used. Keep it bitwise-stable;
tests/test_init_seed.py pins the shared test vectors.
"""

import zlib


def derive_task_seed(base_seed, suite, task_name):
    """Stable per-task seed: independent of generation order, so any subset of
    suites/tasks can be (re)generated in isolation and still match a full run."""
    return (base_seed * 1000003 + zlib.crc32(f"{suite}/{task_name}".encode())) % (2**32)


def mix_counts(num_trials, n_official_avail, n_seeded_avail):
    """Split num_trials into (n_official, n_seeded) episodes.

    Official episodes come first (indices [0, n_official)) so the official
    half of a mixed run replays exactly the same layouts as the first episodes
    of an unmixed run — the two stay directly comparable. Odd trial counts
    favour the official side. If one source has fewer states than its share,
    the other side absorbs the remainder; the total never exceeds what is
    available.
    """
    if num_trials <= 0:
        return 0, 0
    n_official = min((num_trials + 1) // 2, n_official_avail)
    n_seeded = min(num_trials - n_official, n_seeded_avail)
    n_official = min(num_trials - n_seeded, n_official_avail)
    return n_official, n_seeded
