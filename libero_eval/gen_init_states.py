"""Generate seeded init states for LIBERO / LIBERO-Pro suites.

The official benchmarks freeze 50 placement draws per task into .pruned_init
files, so the eval set is fully deterministic — in an adversarial (subnet)
setting a miner can overfit to those exact layouts. Object initial positions
and rotations are the one perturbation dimension that is continuous: every env
reset re-samples them via np.random.uniform over the BDDL region ranges
(LIBERO base_region_sampler). Re-sampling with a post-submission, publicly
derived seed therefore yields a fresh init set from the *same distribution*
that cannot be memorized before submission.

Runs inside the LIBERO client venv (Python 3.8), same contract as
eval_task.py: the caller injects PYTHONPATH (libero package root) and
LIBERO_CONFIG_PATH. For LIBERO-Pro suites the merged bddl/init roots must have
been prepared once (run_eval.py does this via bench.prepare()).

Example (from the validator root, LIBERO-Pro):
    PYTHONPATH=third_party/LIBERO-PRO \
    LIBERO_CONFIG_PATH=~/.cache/libero_pro/config \
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
    third_party/openpi/examples/libero/.venv/bin/python \
        libero_eval/gen_init_states.py --seed 10007 \
        --output-root ~/.cache/libero_pro_custom/init_files_seed10007

Output layout mirrors the official init_files root
({root}/{suite}/{task}.pruned_init, torch.load-compatible), so the whole root
can be passed to run_eval.py / eval_task.py via --init-states-root. A
manifest.json in the root records the base seed and per-task derived seeds.
"""

import argparse
import datetime
import fcntl
import json
import os
import pathlib
import pickle
import time
import zipfile

import numpy as np
from init_mix import derive_task_seed
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# Resolution only affects rendered observations, not the sampled sim state;
# keep it small (matches the upstream LIBERO-PRO generation script).
RESOLUTION = 128


def log(message):
    timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{timestamp} [gen_init_states] {message}", flush=True)


def save_pruned_init(path, states):
    """Write states in the same zip layout as the official .pruned_init files
    (legacy torch serialization: archive/data.pkl holding a pickled ndarray),
    so libero's torch.load(...) reads ours and the official files identically.

    Written to a temp file and renamed: concurrent generators racing on the
    same task (same seed, so identical content) never expose a partial file."""
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("archive/data.pkl", pickle.dumps(states))
        zf.writestr("archive/version", b"1")
    os.replace(str(tmp), str(path))


def load_manifest(path, base_seed, num_inits):
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError:
        manifest = {}
    if manifest.get("base_seed", base_seed) != base_seed or manifest.get("num_inits", num_inits) != num_inits:
        raise SystemExit(
            "{} already holds states for base_seed={} num_inits={}; use a different --output-root".format(
                path.parent, manifest.get("base_seed"), manifest.get("num_inits")
            )
        )
    manifest.setdefault("base_seed", base_seed)
    manifest.setdefault("num_inits", num_inits)
    tasks = manifest.setdefault("tasks", {})
    if not isinstance(tasks, dict):
        raise SystemExit(f"invalid tasks map in {path}")
    return manifest


def save_manifest(path, manifest):
    """Atomically replace a suite manifest while other task workers are active."""
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(str(tmp), str(path))


def parse_task_specs(raw):
    specs = []
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        suite, separator, task_id = item.rpartition(":")
        if not separator or not suite:
            raise SystemExit(f"invalid --task-specs item {item!r}; expected SUITE:TASK_ID")
        try:
            specs.append((suite, int(task_id)))
        except ValueError:
            raise SystemExit(f"invalid task id in --task-specs item {item!r}") from None
    if not specs:
        raise SystemExit("--task-specs must contain at least one SUITE:TASK_ID item")
    if len(specs) != len(set(specs)):
        raise SystemExit("--task-specs contains duplicate suite/task pairs")
    return specs


def generate_for_task(task, num_inits, task_seed):
    bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if not bddl_file.exists():
        raise FileNotFoundError(f"bddl file not found: {bddl_file}")

    np.random.seed(task_seed)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_heights=RESOLUTION, camera_widths=RESOLUTION)
    try:
        # Each reset re-runs the BDDL placement initializer (continuous
        # position/rotation sampling from the np.random stream seeded above).
        # The env constructor already performed one reset; discard it and draw
        # num_inits fresh states so the output depends only on task_seed.
        states = []
        for _ in range(num_inits):
            env.reset()
            states.append(env.get_sim_state())
    finally:
        env.close()
    return np.array(states)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, help="Base seed; per-task seeds are derived from it")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--suites",
        default=None,
        help="Comma-separated suite names registered in the libero package on PYTHONPATH "
        "(default: the validator's 16 LIBERO-Pro perturbation suites)",
    )
    selection.add_argument(
        "--task-specs",
        default=None,
        help="Comma-separated SUITE:TASK_ID pairs; used by run_eval.py for task-level parallel generation",
    )
    parser.add_argument("--num-inits", type=int, default=50, help="Init states per task (official files ship 50)")
    parser.add_argument("--output-root", required=True, help="Root dir for {suite}/{task}.pruned_init output")
    args = parser.parse_args()

    requested_tasks = parse_task_specs(args.task_specs) if args.task_specs else None
    if requested_tasks:
        suites = list(dict.fromkeys(suite for suite, _ in requested_tasks))
    elif args.suites:
        suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    else:
        # Same 16 suites as benchmarks.BENCHMARKS["libero_pro"].default_suites;
        # not imported to keep this script standalone in the client venv.
        bases = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
        dims = ("object", "swap", "lan", "task")
        suites = [f"{b}_{d}" for b in bases for d in dims]

    output_root = pathlib.Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    unknown = [s for s in suites if s not in benchmark_dict]
    if unknown:
        raise SystemExit(f"unknown suites {unknown}; not registered in the libero package on PYTHONPATH")

    task_suites = {suite: benchmark_dict[suite]() for suite in suites}
    if requested_tasks is None:
        work_items = [
            (suite, task_id) for suite, task_suite in task_suites.items() for task_id in range(task_suite.n_tasks)
        ]
    else:
        out_of_range = [
            (suite, task_id)
            for suite, task_id in requested_tasks
            if not 0 <= task_id < task_suites[suite].n_tasks
        ]
        if out_of_range:
            raise SystemExit(f"task ids out of range: {out_of_range}")
        work_items = requested_tasks

    total_tasks = len(work_items)
    completed_tasks = 0
    log(
        f"start seed={args.seed} suites={len(task_suites)} tasks={total_tasks} "
        f"num_inits_per_task={args.num_inits} output_root={output_root}"
    )

    for suite, task_id in work_items:
        task_suite = task_suites[suite]
        suite_dir = output_root / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = suite_dir / "manifest.json"
        manifest_lock_path = suite_dir / ".manifest.lock"
        task_lock_path = suite_dir / f".task{task_id:04d}.lock"

        # A task lock prevents duplicate work when two run_eval processes race
        # on the same cache. Different tasks in one suite still run in parallel;
        # their brief manifest updates are serialized separately below.
        with task_lock_path.open("a+") as task_lock:
            fcntl.flock(task_lock.fileno(), fcntl.LOCK_EX)
            task = task_suite.get_task(task_id)
            out_path = suite_dir / task.init_states_file
            task_seed = derive_task_seed(args.seed, suite, task.name)
            manifest = load_manifest(manifest_path, args.seed, args.num_inits)
            if out_path.exists() and task.name in manifest["tasks"]:
                completed_tasks += 1
                log(
                    f"progress seed={args.seed} tasks={completed_tasks}/{total_tasks} "
                    f"remaining={total_tasks - completed_tasks} suite={suite} task={task_id:02d} status=cached"
                )
                continue
            log(
                f"task start seed={args.seed} suite={suite} task={task_id:02d} "
                f"task_seed={task_seed} completed={completed_tasks}/{total_tasks} "
                f"remaining={total_tasks - completed_tasks}"
            )
            t0 = time.time()
            states = generate_for_task(task, args.num_inits, task_seed)
            save_pruned_init(out_path, states)
            with manifest_lock_path.open("a+") as manifest_lock:
                fcntl.flock(manifest_lock.fileno(), fcntl.LOCK_EX)
                manifest = load_manifest(manifest_path, args.seed, args.num_inits)
                manifest["tasks"][task.name] = {"seed": task_seed, "shape": list(states.shape)}
                manifest["generated_at"] = datetime.datetime.now().astimezone().isoformat()
                save_manifest(manifest_path, manifest)
            completed_tasks += 1
            log(
                f"progress seed={args.seed} tasks={completed_tasks}/{total_tasks} "
                f"remaining={total_tasks - completed_tasks} suite={suite} task={task_id:02d} "
                f"status=generated states={len(states)} shape={list(states.shape)} "
                f"task_elapsed={time.time() - t0:.1f}s output={out_path}"
            )

    log(f"done seed={args.seed} tasks={completed_tasks}/{total_tasks} output_root={output_root}")


if __name__ == "__main__":
    main()
