"""Compare eval runs on official vs privately re-sampled init states.

Validates the anti-overfitting init-state randomization (gen_init_states.py +
run_eval.py --init-states-root) along the two questions that matter:

1. Discrimination: do the two model variants keep their ranking/gap when the
   init states are re-sampled? (total/suite deltas + task-level correlation)
2. Bias: does re-sampling shift a model's success rate beyond sampling noise?
   (two-proportion z-test on pooled episodes)

Pure stdlib (runs with any python3): reads summary.json + results/*.json from
run_eval.py output dirs.

Usage:
    python3 tools/compare_init_randomization.py \
        --run pi05:official:eval_runs/<run_dir> \
        --run pi05:seeded:eval_runs/<run_dir> \
        [--run pi0:official:... --run pi0:seeded:...]

Each --run is MODEL:VARIANT:PATH where VARIANT is 'official' or 'seeded'.
"""

import argparse
import json
import math
import pathlib
import sys


def load_run(path):
    path = pathlib.Path(path)
    summary = json.loads((path / "summary.json").read_text())
    tasks = {}
    for f in sorted((path / "results").glob("*.json")):
        r = json.loads(f.read_text())
        key = (r["task_suite_name"], r["task_id"])
        tasks[key] = {"successes": r["num_successes"], "trials": r["num_trials"]}
    if not tasks:
        sys.exit("no task results under {}".format(path / "results"))
    return {"summary": summary, "tasks": tasks}


def pooled_rate(tasks):
    s = sum(t["successes"] for t in tasks.values())
    n = sum(t["trials"] for t in tasks.values())
    return s, n, (s / n if n else float("nan"))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_proportion_test(s1, n1, s2, n2):
    """Returns (delta, ci95_low, ci95_high, z, p_two_sided) for p1 - p2."""
    p1, p2 = s1 / n1, s2 / n2
    delta = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    pooled = (s1 + s2) / (n1 + n2)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = delta / se_pooled if se_pooled else 0.0
    p = 2 * (1 - norm_cdf(abs(z)))
    return delta, delta - 1.96 * se, delta + 1.96 * se, z, p


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def rank(values):
    """Average ranks for ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    return pearson(rank(xs), rank(ys))


def fmt_pct(x):
    return f"{x:.1%}" if not math.isnan(x) else "n/a"


def compare_variants(model, runs):
    """official vs seeded for one model."""
    off, seed = runs["official"], runs["seeded"]
    print(f"\n## {model}: official vs seeded init states")

    s1, n1, p1 = pooled_rate(off["tasks"])
    s2, n2, p2 = pooled_rate(seed["tasks"])
    delta, lo, hi, z, p = two_proportion_test(s1, n1, s2, n2)
    print(f"total: official {s1}/{n1} = {fmt_pct(p1)}   seeded {s2}/{n2} = {fmt_pct(p2)}")
    print(
        "delta (official - seeded): {:+.1%}  CI95 [{:+.1%}, {:+.1%}]  z={:.2f} p={:.3f}{}".format(
            delta, lo, hi, z, p, "  ** significant **" if p < 0.05 else "  (within noise)"
        )
    )

    common = sorted(set(off["tasks"]) & set(seed["tasks"]))
    missing = set(off["tasks"]) ^ set(seed["tasks"])
    if missing:
        print(f"WARNING: {len(missing)} tasks present in only one run are excluded")
    xs = [off["tasks"][k]["successes"] / off["tasks"][k]["trials"] for k in common]
    ys = [seed["tasks"][k]["successes"] / seed["tasks"][k]["trials"] for k in common]
    print(
        f"task-level SR correlation over {len(common)} tasks: "
        f"pearson={pearson(xs, ys):.3f} spearman={spearman(xs, ys):.3f}"
    )

    suites = sorted({k[0] for k in common})
    print("{:<28}{:>10}{:>10}{:>8}".format("suite", "official", "seeded", "delta"))
    for suite in suites:
        keys = [k for k in common if k[0] == suite]
        so = sum(off["tasks"][k]["successes"] for k in keys)
        no = sum(off["tasks"][k]["trials"] for k in keys)
        ss = sum(seed["tasks"][k]["successes"] for k in keys)
        ns = sum(seed["tasks"][k]["trials"] for k in keys)
        print(
            "{:<28}{:>10}{:>10}{:>+8.1%}".format(
                suite,
                fmt_pct(so / no if no else float("nan")),
                fmt_pct(ss / ns if ns else float("nan")),
                (so / no - ss / ns) if no and ns else float("nan"),
            )
        )
    return {k: (off["tasks"][k], seed["tasks"][k]) for k in common}


def compare_models(models, by_model):
    """Discrimination: does the model gap/ranking survive re-sampling?"""
    print("\n## model discrimination across variants")
    print("{:<10}{:>12}{:>12}".format("model", "official", "seeded"))
    totals = {}
    for m in models:
        row = []
        for v in ("official", "seeded"):
            _, _, p = pooled_rate(by_model[m][v]["tasks"])
            row.append(p)
        totals[m] = row
        print(f"{m:<10}{fmt_pct(row[0]):>12}{fmt_pct(row[1]):>12}")
    if len(models) == 2:
        a, b = models
        for i, v in enumerate(("official", "seeded")):
            print(f"gap {a} - {b} on {v}: {totals[a][i] - totals[b][i]:+.1%}")
        rank_kept = (totals[a][0] - totals[b][0]) * (totals[a][1] - totals[b][1]) > 0
        print("ranking preserved: {}".format("YES" if rank_kept else "NO"))

        # Task-level: does the per-task model gap correlate across variants?
        common = set(by_model[a]["official"]["tasks"])
        for m in models:
            common &= set(by_model[m]["official"]["tasks"]) & set(by_model[m]["seeded"]["tasks"])
        common = sorted(common)
        gaps = {}
        for v in ("official", "seeded"):
            gaps[v] = [
                by_model[a][v]["tasks"][k]["successes"] / by_model[a][v]["tasks"][k]["trials"]
                - by_model[b][v]["tasks"][k]["successes"] / by_model[b][v]["tasks"][k]["trials"]
                for k in common
            ]
        print(
            "task-level model-gap correlation (official vs seeded, {} tasks): pearson={:.3f} spearman={:.3f}".format(
                len(common), pearson(gaps["official"], gaps["seeded"]), spearman(gaps["official"], gaps["seeded"])
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="MODEL:VARIANT:PATH",
        help="run_eval.py output dir labeled as MODEL:VARIANT:PATH (VARIANT: official|seeded)",
    )
    args = parser.parse_args()

    by_model = {}
    for spec in args.run:
        try:
            model, variant, path = spec.split(":", 2)
        except ValueError:
            sys.exit(f"bad --run spec {spec!r}; expected MODEL:VARIANT:PATH")
        if variant not in ("official", "seeded"):
            sys.exit(f"bad variant {variant!r} in {spec!r}; expected official|seeded")
        by_model.setdefault(model, {})[variant] = load_run(path)

    print("# init-state randomization comparison")
    for model, runs in by_model.items():
        for variant, run in runs.items():
            s = run["summary"]
            print(
                "loaded {}:{} <- {} (benchmark={}, trials/task={}, seed={}, init_states_root={})".format(
                    model,
                    variant,
                    s.get("model"),
                    s.get("benchmark"),
                    s.get("num_trials_per_task"),
                    s.get("seed"),
                    s.get("init_states_root"),
                )
            )

    complete = [m for m, runs in by_model.items() if {"official", "seeded"} <= set(runs)]
    for model in complete:
        compare_variants(model, by_model[model])
    if len(complete) >= 2:
        compare_models(complete[:2], by_model)
    elif len(complete) == 1:
        print("\n(one model only — model-discrimination section needs both models)")


if __name__ == "__main__":
    main()
