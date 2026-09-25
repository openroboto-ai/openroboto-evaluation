"""RoboTwin 2.0 benchmark adapter for :mod:`run_eval`.

Simulation and policy implementation stay in pinned upstream checkouts.  This
module owns only reproducible process orchestration and conversion of the
official per-task output into the validator's summary schema.
"""

from __future__ import annotations

import datetime
import http.client
import json
import math
import os
import pathlib
import queue
import re
import shutil
import socket
import subprocess
import threading
import time

from check_model import check_lingbot_data_contract, check_lingbot_vla_v2_model


OFFICIAL_TASKS = (
    "lift_pot",
    "hanging_mug",
    "stack_bowls_three",
    "scan_object",
    "handover_block",
    "click_bell",
    "put_object_cabinet",
    "open_microwave",
    "stack_blocks_three",
    "place_shoe",
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_mic",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "move_stapler_pad",
    "open_laptop",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "press_stapler",
    "rotate_qrcode",
    "shake_bottle_horizontally",
    "shake_bottle",
    "stack_blocks_two",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
    "put_bottles_dustbin",
)
OFFICIAL_EPISODES_PER_TASK = 100
OFFICIAL_REFERENCE = {"demo_clean": 0.9352, "demo_randomized": 0.9280}
OFFICIAL_SUCCESS_COUNT = {"demo_clean": 4676, "demo_randomized": 4640}
# The model card does not report the topology used to produce its table.  These
# values are only the defaults in the upstream launcher at the result-release
# commit, and must not be presented as the topology of the published run.
REFERENCE_RELEASE_COMMIT = "36a9bab235fff53cec13ae4ffd5e7b22e79d0de8"
REFERENCE_RELEASE_LAUNCHER_DEFAULTS = {
    "gpus": 8,
    "policy_servers_per_gpu": 3,
    "policy_servers_total": 24,
    "simulator_clients_per_policy_server": 1,
}
PUBLIC_CLIENT_INTRODUCED_COMMIT = "f0f53dee6580a1b23b04646a36e4ff82bdfaf21b"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_SUCCESS_RE = re.compile(r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>\s*([\d.]+)%")
_MODEL_CARD_RESULT_RE = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*(\d+(?:\.\d+)?)%\s*\|\s*(\d+(?:\.\d+)?)%\s*\|\s*$",
    re.MULTILINE,
)


def select_tasks(task_names: str | None, task_ids: str | None) -> tuple[str, ...]:
    if task_names and task_ids:
        raise ValueError("RoboTwin --tasks and --task-ids are mutually exclusive")
    if task_names:
        requested = [item.strip() for item in task_names.split(",") if item.strip()]
        unknown = sorted(set(requested) - set(OFFICIAL_TASKS))
        if unknown:
            raise ValueError(f"unknown RoboTwin task(s): {', '.join(unknown)}")
        if not requested:
            raise ValueError("--tasks must contain at least one RoboTwin task name")
        return tuple(dict.fromkeys(requested))
    if task_ids:
        try:
            indexes = [int(item) for item in task_ids.split(",") if item != ""]
        except ValueError as exc:
            raise ValueError("RoboTwin --task-ids must be comma-separated integers") from exc
        if not indexes:
            raise ValueError("--task-ids must contain at least one index")
        invalid = [index for index in indexes if not 0 <= index < len(OFFICIAL_TASKS)]
        if invalid:
            raise ValueError(f"RoboTwin task ids out of range 0..{len(OFFICIAL_TASKS) - 1}: {invalid}")
        return tuple(dict.fromkeys(OFFICIAL_TASKS[index] for index in indexes))
    return OFFICIAL_TASKS


def build_server_command(
    lingbot_python: pathlib.Path,
    checkpoint: pathlib.Path,
    port: int,
    use_compile: bool,
) -> list[str]:
    return [
        str(lingbot_python),
        "-m",
        "deploy.lingbot_vla_v2_policy",
        "--model_path",
        str(checkpoint),
        "--use_length",
        "50",
        "--chunk_ret",
        "True",
        "--use_bf16",
        "True",
        "--use_fp32",
        "False",
        "--use_compile",
        str(bool(use_compile)),
        "--port",
        str(port),
    ]


def build_client_command(
    robotwin_python: pathlib.Path,
    client_script: pathlib.Path,
    robotwin_dir: pathlib.Path,
    task: str,
    task_config: str,
    port: int,
    output_dir: pathlib.Path,
    instruction_type: str | None = None,
) -> list[str]:
    command = [
        str(robotwin_python),
        "-u",
        str(client_script),
        "--config",
        str(robotwin_dir / "policy" / "ACT" / "deploy_policy.yml"),
        "--overrides",
        "--task_name",
        task,
        "--task_config",
        task_config,
        "--train_config_name",
        "0",
        "--seed",
        "0",
        "--policy_name",
        "ACT",
        "--port",
        str(port),
        "--robo_name",
        "robotwin",
        "--video_fps",
        "10",
        "--eval_video_log",
        "False",
        "--output_dir",
        str(output_dir),
    ]
    if instruction_type is not None:
        command.extend(("--instruction_type", instruction_type))
    return command


def parse_task_log(text: str) -> tuple[int, int, float]:
    matches = _SUCCESS_RE.findall(_ANSI_RE.sub("", text))
    if not matches:
        raise ValueError("task log has no complete 'Success rate: N/N => P%' record")
    successes, episodes, percent = matches[-1]
    successes, episodes = int(successes), int(episodes)
    rate = float(percent) / 100.0
    if not 0 <= successes <= episodes or episodes <= 0:
        raise ValueError(f"invalid RoboTwin result {successes}/{episodes}")
    if abs(rate - successes / episodes) > 0.001:
        raise ValueError(f"inconsistent RoboTwin result {successes}/{episodes} vs {percent}%")
    return successes, episodes, successes / episodes


def summarize(results: dict[str, dict], selected_tasks: tuple[str, ...], task_config: str) -> dict:
    completed = [row for row in results.values() if row.get("status") == "ok"]
    successes = sum(row["num_successes"] for row in completed)
    episodes = sum(row["num_trials"] for row in completed)
    suite = f"robotwin_{task_config.removeprefix('demo_')}"
    task_rows = {
        task: {
            "status": row.get("status", "failed"),
            "task_suite_name": suite,
            "task_id": OFFICIAL_TASKS.index(task),
            "task_name": task,
            "num_trials": row.get("num_trials", 0),
            "num_successes": row.get("num_successes", 0),
            "success_rate": row.get("success_rate", 0.0),
            "duration_s": row.get("duration_s", 0.0),
            **({"error": row["error"]} if row.get("error") else {}),
        }
        for task, row in results.items()
    }
    return {
        "tasks": task_rows,
        "suites": {
            suite: {
                "tasks": len(results),
                "failed_tasks": len(results) - len(completed),
                "episodes": episodes,
                "successes": successes,
                "success_rate": successes / episodes if episodes else None,
            }
        },
        "total_episodes": episodes,
        "total_successes": successes,
        "total_success_rate": successes / episodes if episodes else None,
        "num_trials_per_task": OFFICIAL_EPISODES_PER_TASK,
        "tasks_selected": list(selected_tasks),
        "task_config": task_config,
    }


def load_published_task_references(checkpoint: pathlib.Path, task_config: str) -> dict[str, float]:
    """Read clean/randomized per-task references from an official checkpoint model card.

    Fine-tuned checkpoints are not required to publish this table, so absence is
    represented by an empty mapping rather than an error.
    """
    column = 1 if task_config == "demo_clean" else 2
    search_dirs = (checkpoint, *checkpoint.parents[:4])
    for directory in search_dirs:
        model_card = directory / "README.md"
        if not model_card.is_file():
            continue
        matches = _MODEL_CARD_RESULT_RE.findall(model_card.read_text(errors="replace"))
        references = {row[0]: float(row[column]) / 100.0 for row in matches}
        if set(OFFICIAL_TASKS).issubset(references):
            return {task: references[task] for task in OFFICIAL_TASKS}
    return {}


def compare_published_task_results(results: dict[str, dict], references: dict[str, float]) -> dict | None:
    """Build an auditable task-level comparison without defining a pass tolerance."""
    comparable = {task: row for task, row in results.items() if row.get("status") == "ok" and task in references}
    if not comparable:
        return None
    tasks = {
        task: {
            "observed_success_rate": row["success_rate"],
            "published_success_rate": references[task],
            "deviation_percentage_points": round((row["success_rate"] - references[task]) * 100, 6),
        }
        for task, row in comparable.items()
    }
    observed = sum(row["num_successes"] for row in comparable.values()) / sum(
        row["num_trials"] for row in comparable.values()
    )
    published = sum(references[task] for task in comparable) / len(comparable)
    deviations = [abs(row["deviation_percentage_points"]) for row in tasks.values()]
    return {
        "tasks_compared": len(comparable),
        "observed_success_rate": observed,
        "published_success_rate": published,
        "aggregate_deviation_percentage_points": round((observed - published) * 100, 6),
        "exact_task_matches": sum(
            row["observed_success_rate"] == row["published_success_rate"] for row in tasks.values()
        ),
        "mean_absolute_task_deviation_percentage_points": round(sum(deviations) / len(deviations), 6),
        "max_absolute_task_deviation_percentage_points": max(deviations),
        "tasks": tasks,
    }


def build_protocol_report(
    results: dict[str, dict],
    selected_tasks: tuple[str, ...],
    task_config: str,
    gpus: list[int],
    clients_per_server: int,
    instruction_type: str | None = None,
) -> dict:
    """Describe protocol completeness separately from published-score equality."""
    effective_instruction_type = instruction_type or "unseen"
    official_request = selected_tasks == OFFICIAL_TASKS and effective_instruction_type == "unseen"
    official_result = bool(
        official_request
        and len(results) == len(OFFICIAL_TASKS)
        and all(
            row.get("status") == "ok" and row.get("num_trials") == OFFICIAL_EPISODES_PER_TASK
            for row in results.values()
        )
    )
    observed_successes = sum(row.get("num_successes", 0) for row in results.values())
    observed_episodes = sum(row.get("num_trials", 0) for row in results.values())
    aggregate_match = observed_successes == OFFICIAL_SUCCESS_COUNT[task_config] if official_result else None
    statistical_comparison = None
    if official_result:
        reference_successes = OFFICIAL_SUCCESS_COUNT[task_config]
        reference_episodes = len(OFFICIAL_TASKS) * OFFICIAL_EPISODES_PER_TASK
        pooled_rate = (observed_successes + reference_successes) / (observed_episodes + reference_episodes)
        standard_error = math.sqrt(pooled_rate * (1 - pooled_rate) * (1 / observed_episodes + 1 / reference_episodes))
        z_score = (
            (observed_successes / observed_episodes) - (reference_successes / reference_episodes)
        ) / standard_error
        p_value = math.erfc(abs(z_score) / math.sqrt(2))
        statistical_comparison = {
            "method": "two-sided two-proportion z-test",
            "alpha": 0.05,
            "independent_binomial_approximation": True,
            "z_score": round(z_score, 6),
            "p_value": round(p_value, 8),
            "statistically_consistent_at_95pct": p_value >= 0.05,
        }
    return {
        "name": f"RoboTwin 2.0 {task_config} 50-task protocol",
        "official_request": official_request,
        # Kept for the worker API: this means protocol-complete, not score-equal.
        "official_result": official_result,
        "expected_task_count": len(OFFICIAL_TASKS),
        "expected_trials_per_task": OFFICIAL_EPISODES_PER_TASK,
        "instruction_type": effective_instruction_type,
        "local_execution_topology": {
            "gpus": len(gpus),
            "policy_servers_total": len(gpus),
            "policy_servers_per_gpu": 1,
            "simulator_clients_per_policy_server": clients_per_server,
            "policy_requests_serialized_per_server": True,
        },
        "published_reference": {
            "success_rate": OFFICIAL_REFERENCE[task_config],
            "successes": OFFICIAL_SUCCESS_COUNT[task_config],
            "episodes": len(OFFICIAL_TASKS) * OFFICIAL_EPISODES_PER_TASK,
            "exact_aggregate_match": aggregate_match,
            "statistical_comparison": statistical_comparison,
            "execution_topology_reported": False,
            "upstream_launcher_defaults_at_release": REFERENCE_RELEASE_LAUNCHER_DEFAULTS,
            "release_commit": REFERENCE_RELEASE_COMMIT,
            "exact_client_publicly_available_at_release": False,
            "public_client_introduced_commit": PUBLIC_CLIENT_INTRODUCED_COMMIT,
        },
    }


def _copy_official_bridge(lingbot_dir: pathlib.Path, robotwin_dir: pathlib.Path) -> pathlib.Path:
    source = lingbot_dir / "experiment" / "robotwin" / "eval_policy_client_lingbotvla.py"
    if not source.is_file():
        raise FileNotFoundError(f"LingBot RoboTwin client is missing: {source}")
    script_dir = robotwin_dir / "script"
    deploy_dir = script_dir / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    destination = script_dir / source.name
    shutil.copy2(source, destination)
    for name in ("__init__.py", "websocket_client_policy.py", "msgpack_numpy.py"):
        helper = lingbot_dir / "deploy" / name
        if not helper.is_file():
            raise FileNotFoundError(f"LingBot deployment helper is missing: {helper}")
        shutil.copy2(helper, deploy_dir / name)
    return destination


def _wait_for_health(process: subprocess.Popen, port: int, timeout: float, log_path: pathlib.Path) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
            raise RuntimeError(f"LingBot server on port {port} exited {process.returncode}:\n{tail}")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/healthz")
            response = conn.getresponse()
            response.read()
            conn.close()
            if response.status == 200:
                return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError(f"LingBot server on port {port} was not ready after {timeout}s (see {log_path})")


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _find_free_ports(base_port: int, count: int) -> list[int]:
    ports = []
    port = base_port
    while len(ports) < count:
        if port > base_port + 200:
            raise RuntimeError(f"No {count} free RoboTwin policy ports found from {base_port}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                port += 1
                continue
        ports.append(port)
        port += 1
    return ports


def build_worker_slots(gpus: list[int], ports: list[int], clients_per_server: int) -> list[tuple[int, int]]:
    if clients_per_server < 1:
        raise ValueError("RoboTwin clients per server must be at least 1")
    return [(gpu, port) for gpu, port in zip(gpus, ports, strict=True) for _ in range(clients_per_server)]


def _emit_progress(progress_file: str | None, detail: dict) -> None:
    if not progress_file:
        return
    path = pathlib.Path(progress_file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"stage": "evaluating", "detail": detail}, sort_keys=True) + "\n")


def _robotwin_environment(robotwin_dir: pathlib.Path, gpu: int) -> dict[str, str]:
    """Build the simulator environment, including setup-time runtime libraries."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
    # websockets 15 honors HTTP_PROXY.  The official bridge connects to
    # ws://0.0.0.0, so explicitly keep all loopback policy traffic local.
    no_proxy = [item for item in env.get("NO_PROXY", env.get("no_proxy", "")).split(",") if item]
    for host in ("localhost", "127.0.0.1", "0.0.0.0"):
        if host not in no_proxy:
            no_proxy.append(host)
    env["NO_PROXY"] = env["no_proxy"] = ",".join(no_proxy)
    runtime_path_file = robotwin_dir / ".runtime_library_path"
    if runtime_path_file.is_file():
        runtime_path = runtime_path_file.read_text().strip()
        if runtime_path:
            current = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = runtime_path + (f":{current}" if current else "")
    return env


def _runtime_preflight(
    lingbot_dir: pathlib.Path,
    robotwin_dir: pathlib.Path,
    lingbot_python: pathlib.Path,
    robotwin_python: pathlib.Path,
    qwen3_path: pathlib.Path,
) -> None:
    required = (
        lingbot_dir / "deploy" / "lingbot_vla_v2_policy.py",
        lingbot_dir / "configs" / "robot_configs" / "robotwin.yaml",
        lingbot_dir / "assets" / "norm_stats" / "robotwin.json",
        robotwin_dir / "policy" / "ACT" / "deploy_policy.yml",
        robotwin_dir / "task_config" / "demo_clean.yml",
        robotwin_dir / "task_config" / "demo_randomized.yml",
        robotwin_dir / "assets" / "embodiments",
        robotwin_dir / "assets" / "objects",
        lingbot_python,
        robotwin_python,
        qwen3_path / "config.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "RoboTwin runtime is incomplete; run setup_robotwin.sh. Missing:\n  " + "\n  ".join(missing)
        )
    bridge_import = subprocess.run(
        [
            str(robotwin_python),
            "-c",
            "import msgpack, websockets.sync.client; "
            "from script.deploy.websocket_client_policy import WebsocketClientPolicy",
        ],
        cwd=robotwin_dir,
        env=_robotwin_environment(robotwin_dir, 0),
        text=True,
        capture_output=True,
    )
    if bridge_import.returncode:
        detail = (bridge_import.stderr or bridge_import.stdout).strip()
        raise RuntimeError(f"RoboTwin policy bridge import failed; rerun setup_robotwin.sh:\n{detail}")


def run(
    args,
    checkpoint: pathlib.Path,
    gpus: list[int],
    lingbot_dir: pathlib.Path,
    robotwin_dir: pathlib.Path,
    lingbot_python: pathlib.Path,
    robotwin_python: pathlib.Path,
    qwen3_path: pathlib.Path,
) -> int:
    selected_tasks = select_tasks(args.tasks, args.task_ids)
    if args.suites:
        raise ValueError("RoboTwin does not use --suites; select names with --tasks or indexes with --task-ids")
    if args.num_trials not in (None, OFFICIAL_EPISODES_PER_TASK):
        raise ValueError(f"the official RoboTwin client requires --num-trials {OFFICIAL_EPISODES_PER_TASK}")
    if not gpus:
        raise ValueError("RoboTwin needs at least one GPU")
    if args.workers_per_gpu > 1:
        print(
            f"[run_eval] RoboTwin concurrency: {args.workers_per_gpu} simulator clients share each "
            "official policy server; inference remains serialized by the upstream WebSocket service"
        )

    check = check_lingbot_vla_v2_model(checkpoint)
    if not args.skip_model_check and not check.ok:
        problems = "\n".join(f"  {i}. {problem}" for i, problem in enumerate(check.errors, 1))
        raise ValueError(f"LingBot-VLA 2.0 checkpoint check failed:\n{problems}")
    if not args.skip_model_check:
        contract_errors = check_lingbot_data_contract(
            checkpoint,
            expected_cameras=("camera_top", "camera_wrist_left", "camera_wrist_right"),
            required_joints={"arm.position": 14, "end.position": 14, "effector.position": 2},
        )
        if contract_errors:
            problems = "\n".join(f"  {index}. {problem}" for index, problem in enumerate(contract_errors, 1))
            raise ValueError(f"checkpoint is not fine-tuned for the official RoboTwin data contract:\n{problems}")

    model_tag = pathlib.Path(str(args.model).rstrip("/")).name.replace("/", "_")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        pathlib.Path(args.output_dir).resolve()
        if args.output_dir
        else pathlib.Path(__file__).resolve().parents[1] / "eval_runs" / f"{stamp}_robotwin_{model_tag}"
    )
    logs_dir = out_dir / "logs"
    results_dir = out_dir / "results"
    sim_output_dir = out_dir / "eval_results"
    for path in (logs_dir, results_dir, sim_output_dir):
        path.mkdir(parents=True, exist_ok=True)

    client_script = robotwin_dir / "script" / "eval_policy_client_lingbotvla.py"
    if not args.dry_run:
        _runtime_preflight(lingbot_dir, robotwin_dir, lingbot_python, robotwin_python, qwen3_path)
        client_script = _copy_official_bridge(lingbot_dir, robotwin_dir)

    ports = _find_free_ports(args.base_port, len(gpus))
    server_commands = [build_server_command(lingbot_python, checkpoint, port, args.lingbot_compile) for port in ports]
    client_commands = {
        task: build_client_command(
            robotwin_python,
            client_script,
            robotwin_dir,
            task,
            args.robotwin_task_config,
            ports[index % len(ports)],
            sim_output_dir,
            args.robotwin_instruction_type,
        )
        for index, task in enumerate(selected_tasks)
    }
    if args.dry_run:
        (out_dir / "commands.json").write_text(
            json.dumps({"servers": server_commands, "clients": client_commands}, indent=2)
        )
        print(f"[run_eval] RoboTwin dry-run commands written to {out_dir / 'commands.json'}")
        return 0

    server_processes = []
    server_logs = []
    env_base = dict(os.environ)
    env_base["QWEN3VL_PATH"] = str(qwen3_path)
    try:
        for gpu, port, command in zip(gpus, ports, server_commands, strict=True):
            log_path = logs_dir / f"server_gpu{gpu}_port{port}.log"
            log_file = log_path.open("w")
            process = subprocess.Popen(
                command,
                cwd=lingbot_dir,
                env={**env_base, "CUDA_VISIBLE_DEVICES": str(gpu)},
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            server_processes.append(process)
            server_logs.append(log_file)
            _wait_for_health(process, port, args.server_timeout, log_path)
            print(f"[run_eval] LingBot-VLA 2.0 server ready: gpu={gpu} port={port}")

        work: queue.Queue[tuple[int, str]] = queue.Queue()
        for index, task in enumerate(selected_tasks):
            work.put((index, task))
        results: dict[str, dict] = {}
        lock = threading.Lock()
        progress = {"done": 0}
        _emit_progress(
            args.progress_file,
            {
                "tasks_done": 0,
                "tasks_total": len(selected_tasks),
                "episodes_done": 0,
                "episodes_total": len(selected_tasks) * OFFICIAL_EPISODES_PER_TASK,
            },
        )

        def worker(slot: int, gpu: int, port: int) -> None:
            while True:
                try:
                    _, task = work.get_nowait()
                except queue.Empty:
                    return
                log_path = logs_dir / f"robotwin_{task}.log"
                result_path = results_dir / f"robotwin_{task}.json"
                if args.resume and result_path.is_file():
                    try:
                        previous = json.loads(result_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        previous = None
                    if isinstance(previous, dict) and previous.get("status") == "ok":
                        with lock:
                            results[task] = previous
                            progress["done"] += 1
                            _emit_progress(
                                args.progress_file,
                                {
                                    "tasks_done": progress["done"],
                                    "tasks_total": len(selected_tasks),
                                    "episodes_done": progress["done"] * OFFICIAL_EPISODES_PER_TASK,
                                    "episodes_total": len(selected_tasks) * OFFICIAL_EPISODES_PER_TASK,
                                },
                            )
                        work.task_done()
                        continue

                command = build_client_command(
                    robotwin_python,
                    client_script,
                    robotwin_dir,
                    task,
                    args.robotwin_task_config,
                    port,
                    sim_output_dir,
                    args.robotwin_instruction_type,
                )
                started = time.time()
                row = None
                for attempt in range(args.retries + 1):
                    with log_path.open("a") as log_file:
                        log_file.write(f"\n===== attempt {attempt + 1} gpu={gpu} port={port} =====\n")
                        log_file.flush()
                        try:
                            completed = subprocess.run(
                                command,
                                cwd=robotwin_dir,
                                env=_robotwin_environment(robotwin_dir, gpu),
                                stdout=log_file,
                                stderr=subprocess.STDOUT,
                                timeout=args.task_timeout or 8 * 3600,
                            )
                            returncode = completed.returncode
                        except subprocess.TimeoutExpired:
                            returncode = -1
                    if returncode == 0:
                        try:
                            successes, episodes, rate = parse_task_log(log_path.read_text(errors="replace"))
                            if episodes != OFFICIAL_EPISODES_PER_TASK:
                                raise ValueError(
                                    f"task completed {episodes}/{OFFICIAL_EPISODES_PER_TASK} official episodes"
                                )
                            row = {
                                "status": "ok",
                                "num_successes": successes,
                                "num_trials": episodes,
                                "success_rate": rate,
                                "duration_s": round(time.time() - started, 1),
                            }
                            break
                        except ValueError as exc:
                            error = str(exc)
                    else:
                        error = f"client exited {returncode}"
                    row = {"status": "failed", "error": error, "duration_s": round(time.time() - started, 1)}
                assert row is not None
                result_path.write_text(json.dumps(row, indent=2))
                with lock:
                    results[task] = row
                    progress["done"] += 1
                    print(f"[run_eval] [{progress['done']}/{len(selected_tasks)}] RoboTwin {task}: {row['status']}")
                    _emit_progress(
                        args.progress_file,
                        {
                            "tasks_done": progress["done"],
                            "tasks_total": len(selected_tasks),
                            "episodes_done": progress["done"] * OFFICIAL_EPISODES_PER_TASK,
                            "episodes_total": len(selected_tasks) * OFFICIAL_EPISODES_PER_TASK,
                            "last_completed_task": task,
                        },
                    )
                work.task_done()

        started = time.time()
        worker_slots = build_worker_slots(gpus, ports, args.workers_per_gpu)
        threads = [
            threading.Thread(target=worker, args=(slot, gpu, port), daemon=True)
            for slot, (gpu, port) in enumerate(worker_slots)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        summary = summarize(results, selected_tasks, args.robotwin_task_config)
        published_references = load_published_task_references(checkpoint, args.robotwin_task_config)
        published_comparison = compare_published_task_results(results, published_references)
        if published_comparison is not None:
            summary["published_task_comparison"] = published_comparison
        protocol = build_protocol_report(
            results,
            selected_tasks,
            args.robotwin_task_config,
            gpus,
            args.workers_per_gpu,
            args.robotwin_instruction_type,
        )
        summary.update({
            "model": args.model,
            "commit_id": args.commit_id,
            "checkpoint_dir": str(checkpoint),
            "benchmark": "robotwin",
            "backbone": "lingbot-vla-v2",
            "model_family": "lingbot_vla_v2",
            "gpus": gpus,
            "simulator_clients_per_policy_server": args.workers_per_gpu,
            "instruction_type": args.robotwin_instruction_type or "unseen",
            "wall_time_s": round(time.time() - started, 1),
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "evaluation_protocol": protocol,
        })
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        rate = summary["total_success_rate"]
        print(f"[run_eval] RoboTwin summary: {rate:.2%}" if rate is not None else "[run_eval] no completed tasks")
        print(f"[run_eval] summary written to {out_dir / 'summary.json'}")
        return 0 if all(row.get("status") == "ok" for row in results.values()) else 2
    finally:
        for process in server_processes:
            _terminate(process)
        for log_file in server_logs:
            log_file.close()
