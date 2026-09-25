"""RoboDojo backend for :mod:`run_eval`.

The simulator and policy adapters remain owned by the pinned RoboDojo checkout.
This module only selects the canonical protocol, launches its public server/client
entry points, and converts `_result.json` files into validator summaries.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
import os
import pathlib
import queue
import signal
import socket
import subprocess
import threading
import time

from check_model import check_model


DIMENSIONS = {
    "generalization": (
        "stack_bowls",
        "push_T",
        "pack_objects_into_box",
        "fold_clothes",
        "hang_mugs",
        "sweep_blocks",
        "pour_liquid_into_cup",
        "make_toast",
        "arrange_largest_number",
        "sort_nesting_dolls_by_size",
        "store_laptop_and_headphones",
        "stack_blocks",
    ),
    "precision": (
        "fasten_screws",
        "plug_in_charger",
        "insert_tubes",
        "pour_balls_into_vase",
        "play_Xylophone",
        "deposit_coin",
        "insert_key",
        "build_tower",
    ),
    "long-horizon": (
        "put_bottles_into_dustbin",
        "fill_pen_holder",
        "classify_objects",
        "play_tic_tac_toe",
        "fill_egg_holder",
        "organize_table",
        "make_kong",
        "play_stacking_toy",
    ),
    "memory": (
        "cover_blocks",
        "match_and_pick_from_conveyor",
        "swap_blocks",
        "swap_T",
        "press_by_number",
        "imitate_sorting_sequence",
    ),
    "open": (
        "align_blocks",
        "general_pickup",
        "stack_blocks_by_language",
        "solve_equation",
        "classify_objects_by_language",
        "pick_from_conveyor_by_image",
        "store_tools_in_toolbox",
        "pour_by_language",
    ),
}
GENERALIZATION = frozenset(DIMENSIONS["generalization"])
OFFICIAL_SEEDS = (0, 1, 2)
OFFICIAL_STANDALONE_EPISODES = 50
OFFICIAL_GENERALIZATION_HALF_EPISODES = 25


@dataclasses.dataclass(frozen=True)
class RunSpec:
    dimension: str
    report_task: str
    sim_task: str
    eval_seed: int

    @property
    def key(self) -> str:
        return f"seed{self.eval_seed}_{self.sim_task}"


def parse_dimensions(value: str | None) -> tuple[str, ...]:
    if value is None:
        return tuple(DIMENSIONS)
    aliases = {"long_horizon": "long-horizon", "longhorizon": "long-horizon"}
    selected = []
    for raw in value.split(","):
        name = aliases.get(raw.strip().lower(), raw.strip().lower())
        if name not in DIMENSIONS:
            raise ValueError(f"unknown RoboDojo dimension {raw!r}; choose from {', '.join(DIMENSIONS)}")
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError("at least one RoboDojo dimension is required")
    return tuple(selected)


def select_tasks(dimensions: tuple[str, ...], value: str | None) -> tuple[str, ...]:
    available = [task for dim in dimensions for task in DIMENSIONS[dim]]
    if value is None:
        return tuple(available)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(
            f"unknown task(s) for selected RoboDojo dimensions: {', '.join(unknown)}; "
            "pass base task names (the generalization _random half is scheduled automatically)"
        )
    return tuple(dict.fromkeys(requested))


def build_specs(tasks: tuple[str, ...], eval_seeds: tuple[int, ...]) -> list[RunSpec]:
    task_dimension = {task: dim for dim, names in DIMENSIONS.items() for task in names}
    specs = []
    for seed in eval_seeds:
        for task in tasks:
            specs.append(RunSpec(task_dimension[task], task, task, seed))
            if task in GENERALIZATION:
                specs.append(RunSpec(task_dimension[task], task, f"{task}_random", seed))
    return specs


def validate_checkpoint(ckpt_dir: pathlib.Path, architectures: tuple[str, ...]):
    results = []
    for architecture in architectures:
        config = "pi05_robodojo" if architecture == "pi0.5" else "pi0_robodojo"
        result = check_model(ckpt_dir, config)
        if result.ok:
            return architecture, config, result
        results.append((architecture, config, result))
    architecture, config, result = min(results, key=lambda item: len(item[2].errors))
    return architecture, config, result


def _policy_name(architecture: str) -> str:
    return "Pi_05" if architecture == "pi0.5" else "Pi_0"


def build_server_command(
    robodojo_dir: pathlib.Path,
    policy_name: str,
    spec: RunSpec,
    ckpt_dir: pathlib.Path,
    gpu: int,
    port: int,
) -> list[str]:
    return [
        "bash",
        str(robodojo_dir / "scripts" / "robodojo.sh"),
        "server",
        "--policy-dir",
        str(robodojo_dir / "XPolicyLab" / "policy" / policy_name),
        "--task",
        spec.sim_task,
        "--ckpt",
        str(ckpt_dir),
        "--env-cfg",
        "arx_x5",
        "--action-type",
        "joint",
        "--seed",
        str(spec.eval_seed),
        "--policy-gpu",
        str(gpu),
        "--policy-env",
        "uv",
        "--policy-port",
        str(port),
        "--bind-host",
        "127.0.0.1",
    ]


def build_client_command(
    robodojo_dir: pathlib.Path,
    policy_name: str,
    spec: RunSpec,
    checkpoint_label: str,
    gpu: int,
    port: int,
    num_trials: int | None,
) -> list[str]:
    cmd = [
        # The public split-client entry point assumes its caller already
        # activated the simulator environment (unlike the monolithic `eval`
        # entry point). Make that precondition explicit and deterministic.
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "RoboDojo",
        "bash",
        str(robodojo_dir / "scripts" / "robodojo.sh"),
        "client",
        "--task",
        spec.sim_task,
        "--policy-name",
        policy_name,
        "--policy-host",
        "127.0.0.1",
        "--policy-port",
        str(port),
        "--env-cfg",
        "arx_x5",
        "--seed",
        str(spec.eval_seed),
        "--env-gpu",
        str(gpu),
        "--ckpt",
        checkpoint_label,
        "--action-type",
        "joint",
    ]
    if num_trials is not None:
        cmd += ["--eval-num", str(num_trials)]
    return cmd


def expected_result_path(
    robodojo_dir: pathlib.Path,
    policy_name: str,
    spec: RunSpec,
    checkpoint_label: str,
    run_id: str,
) -> pathlib.Path:
    return (
        robodojo_dir
        / "eval_result"
        / "RoboDojo"
        / spec.sim_task
        / policy_name
        / "arx_x5"
        / f"{spec.eval_seed}_ckpt_name={checkpoint_label},action_type=joint"
        / run_id
        / "_result.json"
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_server(proc: subprocess.Popen, port: int, log_path: pathlib.Path, timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
            raise RuntimeError(f"policy server exited with code {proc.returncode}:\n{tail}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"policy server did not listen on port {port} within {timeout:.0f}s")


def _stop_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def _entries(payload: dict) -> list[dict]:
    rows = []
    for layout, entry in (payload.get("details") or {}).items():
        try:
            layout_id = int(layout)
        except (TypeError, ValueError):
            continue
        rows.append({
            "layout_id": layout_id,
            "success": bool(entry.get("success", False)),
            "score": float(entry.get("score", 0.0) or 0.0),
        })
    return sorted(rows, key=lambda row: row["layout_id"])


def _run_one(
    spec: RunSpec,
    gpu: int,
    args,
    robodojo_dir: pathlib.Path,
    ckpt_dir: pathlib.Path,
    policy_name: str,
    checkpoint_label: str,
    out_dir: pathlib.Path,
) -> dict:
    run_id = f"{out_dir.name}_{spec.key}"
    result_path = expected_result_path(robodojo_dir, policy_name, spec, checkpoint_label, run_id)
    server_log = out_dir / "logs" / f"{spec.key}_server.log"
    client_log = out_dir / "logs" / f"{spec.key}_client.log"
    port = _free_port()
    server_cmd = build_server_command(robodojo_dir, policy_name, spec, ckpt_dir, gpu, port)
    client_cmd = build_client_command(robodojo_dir, policy_name, spec, checkpoint_label, gpu, port, args.num_trials)
    if args.dry_run:
        return {
            "status": "dry_run",
            "dimension": spec.dimension,
            "report_task": spec.report_task,
            "sim_task": spec.sim_task,
            "eval_seed": spec.eval_seed,
            "server_command": server_cmd,
            "client_command": client_cmd,
            "expected_result_path": str(result_path),
        }

    env = dict(os.environ)
    env.update({
        "ROBODOJO_RUN_ID": run_id,
        "ROBODOJO_FATAL_RESTART_COUNT": "0",
        "OMNI_KIT_ACCEPT_EULA": "YES",
    })
    started = time.monotonic()
    server = None
    try:
        with server_log.open("w") as server_f:
            server = subprocess.Popen(
                server_cmd,
                cwd=robodojo_dir,
                env=env,
                stdout=server_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _wait_server(server, port, server_log, timeout=args.server_timeout)
            with client_log.open("w") as client_f:
                completed = subprocess.run(
                    client_cmd,
                    cwd=robodojo_dir,
                    env=env,
                    stdout=client_f,
                    stderr=subprocess.STDOUT,
                    timeout=args.task_timeout or None,
                )
        if completed.returncode != 0:
            raise RuntimeError(f"simulator client exited with code {completed.returncode}; see {client_log}")
        if not result_path.is_file():
            raise RuntimeError(f"simulator exited successfully but did not write {result_path}")
        payload = json.loads(result_path.read_text())
        entries = _entries(payload)
        return {
            "status": "ok",
            "dimension": spec.dimension,
            "report_task": spec.report_task,
            "sim_task": spec.sim_task,
            "eval_seed": spec.eval_seed,
            "num_trials": len(entries),
            "num_successes": sum(row["success"] for row in entries),
            "success_rate": sum(row["success"] for row in entries) / len(entries) if entries else None,
            "score": sum(row["score"] for row in entries) / len(entries) if entries else None,
            "eval_time": payload.get("eval_time"),
            "wall_time_s": round(time.monotonic() - started, 1),
            "episodes": entries,
            "source_result": str(result_path),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "dimension": spec.dimension,
            "report_task": spec.report_task,
            "sim_task": spec.sim_task,
            "eval_seed": spec.eval_seed,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_time_s": round(time.monotonic() - started, 1),
        }
    finally:
        if server is not None:
            _stop_process_group(server)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) if values else None


def summarize(raw_results: dict, tasks: tuple[str, ...], seeds: tuple[int, ...], num_trials: int | None) -> dict:
    by_run = {(row.get("sim_task"), row.get("eval_seed")): row for row in raw_results.values()}
    cells = {}
    for dimension, dimension_tasks in DIMENSIONS.items():
        for task in dimension_tasks:
            if task not in tasks:
                continue
            for seed in seeds:
                base = by_run.get((task, seed), {})
                base_entries = base.get("episodes", []) if base.get("status") == "ok" else []
                sources = [task]
                required = num_trials if num_trials is not None else OFFICIAL_STANDALONE_EPISODES
                entries = base_entries[:required]
                if task in GENERALIZATION:
                    random_task = f"{task}_random"
                    random = by_run.get((random_task, seed), {})
                    random_entries = random.get("episodes", []) if random.get("status") == "ok" else []
                    half = num_trials if num_trials is not None else OFFICIAL_GENERALIZATION_HALF_EPISODES
                    entries = base_entries[:half] + random_entries[:half]
                    required = half * 2
                    sources.append(random_task)
                complete = len(entries) >= required
                key = f"{task}:seed{seed}"
                cells[key] = {
                    "dimension": dimension,
                    "task": task,
                    "eval_seed": seed,
                    "status": "ok" if complete else "incomplete",
                    "source_tasks": sources,
                    "episodes": len(entries),
                    "required_episodes": required,
                    "success_rate": _mean([float(row["success"]) for row in entries]) if entries else None,
                    "score": _mean([row["score"] for row in entries]) if entries else None,
                }

    dimensions = {}
    per_seed_overall = {seed: [] for seed in seeds}
    for dimension in DIMENSIONS:
        per_seed = {}
        for seed in seeds:
            dimension_cells = [
                cell
                for cell in cells.values()
                if cell["dimension"] == dimension and cell["eval_seed"] == seed and cell["status"] == "ok"
            ]
            if dimension_cells:
                per_seed[seed] = {
                    "success_rate": _mean([cell["success_rate"] for cell in dimension_cells]),
                    "score": _mean([cell["score"] for cell in dimension_cells]),
                    "completed_tasks": len(dimension_cells),
                }
                per_seed_overall[seed].append(per_seed[seed])
        sr_values = [row["success_rate"] for row in per_seed.values()]
        score_values = [row["score"] for row in per_seed.values()]
        dimensions[dimension] = {
            "success_rate": _mean(sr_values),
            "success_rate_std": _std(sr_values),
            "score": _mean(score_values),
            "score_std": _std(score_values),
            "seeds": per_seed,
        }
    seed_overall = {
        seed: {
            "success_rate": _mean([row["success_rate"] for row in values]),
            "score": _mean([row["score"] for row in values]),
        }
        for seed, values in per_seed_overall.items()
        if values
    }
    completed = sum(cell["status"] == "ok" for cell in cells.values())
    failed_runs = sum(row.get("status") == "failed" for row in raw_results.values())
    return {
        "dimensions": dimensions,
        "task_seed_results": cells,
        "completed_task_seed_cells": completed,
        "expected_task_seed_cells": len(tasks) * len(seeds),
        "failed_simulator_runs": failed_runs,
        "per_seed": seed_overall,
        "total_success_rate": _mean([row["success_rate"] for row in seed_overall.values()]),
        "total_score": _mean([row["score"] for row in seed_overall.values()]),
        "raw_runs": dict(sorted(raw_results.items())),
    }


def _runtime_preflight(robodojo_dir: pathlib.Path, policy_name: str) -> None:
    required = [
        robodojo_dir / "scripts" / "robodojo.sh",
        robodojo_dir / "XPolicyLab" / "policy" / policy_name / "setup_eval_policy_server.sh",
        robodojo_dir / "XPolicyLab" / "policy" / policy_name / "setup_eval_env_client.sh",
        robodojo_dir / "XPolicyLab" / "policy" / policy_name / "openpi" / ".venv" / "bin" / "python",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("RoboDojo runtime is not installed; missing:\n  " + "\n  ".join(missing))
    doctor = subprocess.run(
        ["bash", str(robodojo_dir / "scripts" / "robodojo.sh"), "doctor"],
        cwd=robodojo_dir,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
        capture_output=True,
        text=True,
    )
    if doctor.returncode:
        raise RuntimeError(f"RoboDojo doctor failed:\n{doctor.stdout}\n{doctor.stderr}")


def run(
    args,
    checkpoints: pathlib.Path | dict[int, pathlib.Path],
    architectures: tuple[str, ...],
    gpus: list[int],
    robodojo_dir: pathlib.Path,
):
    dimensions = parse_dimensions(args.suites)
    tasks = select_tasks(dimensions, args.tasks)
    try:
        seeds = tuple(dict.fromkeys(int(item) for item in args.eval_seeds.split(",") if item != ""))
    except ValueError as exc:
        raise ValueError("--eval-seeds must be comma-separated integers") from exc
    if not seeds:
        raise ValueError("--eval-seeds must contain at least one seed")
    if args.num_trials is not None and args.num_trials < 1:
        raise ValueError("--num-trials must be at least 1")
    if args.task_ids is not None:
        raise ValueError("--task-ids is not used by RoboDojo; select names with --tasks")
    if args.workers_per_gpu != 1:
        print("[run_eval] note: RoboDojo always runs one simulator per GPU; --workers-per-gpu is ignored")

    ckpt_by_seed = checkpoints if isinstance(checkpoints, dict) else {seed: checkpoints for seed in seeds}
    if set(ckpt_by_seed) != set(seeds):
        raise ValueError(f"checkpoint seeds {sorted(ckpt_by_seed)} do not match evaluation seeds {list(seeds)}")
    checked = {seed: validate_checkpoint(path, architectures) for seed, path in ckpt_by_seed.items()}
    architecture, config, _ = checked[seeds[0]]
    for seed, (seed_architecture, seed_config, check) in checked.items():
        if not args.skip_model_check and not check.ok:
            problems = "\n".join(f"  {i}. {problem}" for i, problem in enumerate(check.errors, 1))
            raise ValueError(f"RoboDojo seed {seed} {seed_architecture} checkpoint check failed:\n{problems}")
        if (seed_architecture, seed_config) != (architecture, config):
            raise ValueError("all RoboDojo seed checkpoints must use the same model architecture")
    if args.config is not None and args.config != config:
        raise ValueError(f"RoboDojo selected config is {config!r}, not requested {args.config!r}")
    policy_name = _policy_name(architecture)
    if not args.dry_run:
        _runtime_preflight(robodojo_dir, policy_name)

    model_tag = pathlib.Path(str(args.model).rstrip("/")).name.replace("/", "_")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        pathlib.Path(args.output_dir)
        if args.output_dir
        else (pathlib.Path(__file__).resolve().parent.parent / "eval_runs" / f"{stamp}_robodojo_{model_tag}")
    )
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    specs = build_specs(tasks, seeds)
    pending = []
    results = {}
    for spec in specs:
        result_file = out_dir / "results" / f"{spec.key}.json"
        if args.resume and result_file.is_file():
            previous = json.loads(result_file.read_text())
            if previous.get("status") == "ok":
                results[spec.key] = previous
                continue
        pending.append(spec)

    print(
        f"[run_eval] RoboDojo: {len(tasks)} reported tasks, {len(specs)} simulator runs, "
        f"seeds={list(seeds)}, GPUs={gpus}, checkpoints="
        f"{ {seed: str(path) for seed, path in ckpt_by_seed.items()} }"
    )
    work = queue.Queue()
    for spec in pending:
        work.put(spec)
    lock = threading.Lock()
    progress = {"done": len(results)}

    def worker(gpu):
        while True:
            try:
                spec = work.get_nowait()
            except queue.Empty:
                return
            ckpt_dir = ckpt_by_seed[spec.eval_seed]
            checkpoint_label = f"run_eval_{policy_name.lower()}_seed{spec.eval_seed}_{ckpt_dir.name}"
            for attempt in range(args.retries + 1):
                row = _run_one(spec, gpu, args, robodojo_dir, ckpt_dir, policy_name, checkpoint_label, out_dir)
                if row["status"] != "failed" or attempt == args.retries:
                    break
                print(
                    f"[run_eval] {spec.key}: attempt {attempt + 1} failed; "
                    f"retrying ({row.get('error', 'unknown error')})"
                )
            (out_dir / "results" / f"{spec.key}.json").write_text(json.dumps(row, indent=2))
            with lock:
                results[spec.key] = row
                progress["done"] += 1
                print(f"[run_eval] [{progress['done']}/{len(specs)}] {spec.key}: {row['status']}")
            work.task_done()

    started = time.time()
    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    summary = summarize(results, tasks, seeds, args.num_trials)
    official_request = (
        dimensions == tuple(DIMENSIONS)
        and tasks == tuple(task for names in DIMENSIONS.values() for task in names)
        and seeds == OFFICIAL_SEEDS
        and args.num_trials is None
    )
    official_result = bool(
        official_request
        and summary["completed_task_seed_cells"] == summary["expected_task_seed_cells"] == 126
        and summary["failed_simulator_runs"] == 0
    )
    summary.update({
        "model": args.model,
        "commit_id": args.commit_id,
        "checkpoint_dir": str(ckpt_by_seed[seeds[0]]) if len(set(ckpt_by_seed.values())) == 1 else None,
        "checkpoint_dirs_by_seed": {str(seed): str(path) for seed, path in ckpt_by_seed.items()},
        "benchmark": "robodojo",
        "model_family": "openpi",
        "architecture": architecture,
        "config": config,
        "policy_name": policy_name,
        "dimensions_selected": list(dimensions),
        "tasks_selected": list(tasks),
        "eval_seeds": list(seeds),
        "num_trials_per_sim_task": args.num_trials,
        "gpus": gpus,
        "wall_time_s": round(time.time() - started, 1),
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "evaluation_protocol": {
            "name": "RoboDojo official 42-task/3-seed protocol",
            "official_request": official_request,
            "official_result": official_result,
            "generalization_episodes": "25 standard + 25 random"
            if args.num_trials is None
            else f"{args.num_trials} + {args.num_trials}",
            "standalone_episodes": 50 if args.num_trials is None else args.num_trials,
        },
    })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    rate = summary["total_success_rate"]
    score = summary["total_score"]
    print(
        f"[run_eval] RoboDojo summary: SR={rate:.2%} score={score * 100:.2f}"
        if rate is not None
        else "[run_eval] no completed RoboDojo cells"
    )
    print(f"[run_eval] summary written to {out_dir / 'summary.json'}")
    return 0 if args.dry_run or summary["failed_simulator_runs"] == 0 else 2
