"""Axis benchmark orchestration for :mod:`run_eval`."""

from __future__ import annotations

import datetime
import concurrent.futures
import json
import logging
import os
import pathlib
import shutil
import subprocess
import time
from typing import Any, Callable

from axis_runtime import (
    AXIS_V1_NAME,
    AXIS_V1_CONFIG_PATH,
    canonical_json_sha256,
    load_manifest,
    task_specs,
)
from axis_sampling import sample_tasks
from axis_progress import report_axis_progress
from axis_model_input import AXIS_PI05_DISCRETE_STATE_INPUT, model_uses_discrete_state


AXIS_TASK_SCRIPT = pathlib.Path(__file__).resolve().parent / "axis_task.py"
VALIDATOR_ROOT = pathlib.Path(__file__).resolve().parents[1]
AXIS_BENCHMARKS = (AXIS_V1_NAME, "axis")
AXIS_RENDERER_BACKEND = "osmesa"


def _manifest_path(benchmark: str, override: str | None = None) -> pathlib.Path:
    if benchmark == "axis":
        if not override:
            raise ValueError("--benchmark axis requires --axis-manifest pointing to a frozen versioned manifest")
        return pathlib.Path(override).expanduser().resolve()
    if override is not None:
        raise ValueError("--axis-manifest requires --benchmark axis; named releases cannot be overridden")
    if benchmark not in AXIS_BENCHMARKS:
        raise ValueError(f"unsupported Axis benchmark {benchmark!r}; choose from {list(AXIS_BENCHMARKS)}")
    return AXIS_V1_CONFIG_PATH


def apply_manifest_defaults(args: Any) -> None:
    """Resolve benchmark defaults before downloads, GPU reservations or model checks."""
    manifest = load_manifest(
        _manifest_path(args.benchmark, args.axis_manifest),
        expected_name=None if args.benchmark == "axis" else args.benchmark,
    )
    args.axis_loaded_manifest = manifest
    if getattr(args, "axis_sample_size", None) is not None or getattr(args, "axis_sampling_seed", None) is not None:
        # Freeze the draw before downloads, policy startup and any model scores.
        _selected_task_ids(args, manifest)
    if args.axis_replan_steps is None:
        args.axis_replan_steps = manifest["protocol"].get("replan_steps", 5)
    if args.seed is None:
        args.seed = manifest.get("policy_seed", 7)
    if args.config is None:
        args.config = "pi05_axis_joint"


def _checkpoint_provenance(checkpoint: pathlib.Path | None) -> dict[str, Any] | None:
    """Optional training provenance; it does not control or gate evaluation."""
    if checkpoint is None:
        return None
    metadata_path = checkpoint / "axis_vla_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
    except (OSError, UnicodeError, ValueError):
        logging.warning("Ignoring unreadable optional AXIS training provenance: %s", metadata_path)
        return None
    fields = (
        "artifact_sha256",
        "backbone",
        "checkpoint_kind",
        "config",
        "eligible_for_scoring",
        "global_step",
        "openpi_git_commit",
        "purpose",
        "seed",
        "training_scope",
        "validator_git_commit",
    )
    result = {field: metadata[field] for field in fields if field in metadata}
    if metadata.get("config") == "pi05_axis_joint":
        try:
            result["discrete_state_input"] = model_uses_discrete_state(metadata)
        except ValueError:
            logging.warning("Ignoring invalid state-input training provenance: %s", metadata_path)
    return result


def _selected_task_ids(args: Any, manifest: dict[str, Any]) -> list[int]:
    sample_size = getattr(args, "axis_sample_size", None)
    sampling_seed = getattr(args, "axis_sampling_seed", None)
    if sampling_seed is not None and sample_size is None:
        raise ValueError("--axis-sampling-seed requires --axis-sample-size")
    if sample_size is not None:
        if args.task_ids:
            raise ValueError("--axis-sample-size cannot be combined with --task-ids")
        selection = sample_tasks(manifest, sample_size, sampling_seed)
        args.axis_sampling_seed = selection["sampling_seed"]
        args.axis_task_selection = selection
        return selection["selected_task_ids"]
    available = task_specs(manifest)
    selected = [int(value) for value in args.task_ids.split(",") if value] if args.task_ids else list(available)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"{manifest['name']} has no task ids {unknown}; choose from {sorted(available)}")
    if len(selected) != len(set(selected)):
        raise ValueError(f"--task-ids contains duplicates: {selected}")
    return selected


def _task_command(
    args: Any,
    axis_python: pathlib.Path,
    task_id: int,
    result_path: pathlib.Path,
    *,
    port: int,
    dry_run: bool,
    prepare_only: bool = False,
) -> list[str]:
    command = [
        str(axis_python),
        str(AXIS_TASK_SCRIPT),
        "--task-id",
        str(task_id),
        "--manifest",
        str(
            getattr(args, "axis_resolved_manifest", None)
            or _manifest_path(args.benchmark, getattr(args, "axis_manifest", None))
        ),
        "--cache-root",
        str(pathlib.Path(args.axis_cache_root).expanduser().resolve()),
        "--asset-fetch-workers",
        str(args.axis_asset_fetch_workers),
        "--host",
        args.axis_policy_host,
        "--port",
        str(port),
        "--num-trials",
        str(args.num_trials),
        "--replan-steps",
        str(args.axis_replan_steps),
        "--policy-seed",
        str(args.seed),
        "--result-path",
        str(result_path),
    ]
    if args.axis_task_api_base_url:
        command += ["--task-api-base-url", args.axis_task_api_base_url]
    if args.axis_asset_base_url:
        command += ["--asset-base-url", args.axis_asset_base_url]
    if args.axis_max_control_steps:
        command += ["--max-control-steps", str(args.axis_max_control_steps)]
    if getattr(args, "axis_record_trials", 0):
        command += ["--record-trials", str(args.axis_record_trials)]
    if getattr(args, "axis_randomization_manifest", None):
        command += ["--randomization-manifest", args.axis_randomization_manifest]
    if getattr(args, "axis_randomization_seed", None) is not None:
        command += ["--randomization-seed", str(args.axis_randomization_seed)]
    if prepare_only:
        command.append("--prepare-only")
    if dry_run:
        command.append("--dry-run")
    return command


def _run_task(
    command: list[str],
    log_path: pathlib.Path,
    result_path: pathlib.Path,
    gpu: int,
    timeout_s: float,
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MUJOCO_GL": AXIS_RENDERER_BACKEND,
    })
    environment.pop("MUJOCO_EGL_DEVICE_ID", None)
    # websockets 15+ honors HTTP_PROXY automatically.  A developer shell may
    # have a global proxy without NO_PROXY, which would route the local policy
    # connection through that proxy and fail the WebSocket handshake.  Local
    # policy transports never belong on an HTTP proxy; make the evaluator
    # self-contained instead of relying on a systemd-only environment setting.
    no_proxy = [value.strip() for value in environment.get("NO_PROXY", "").split(",") if value.strip()]
    for local_host in ("127.0.0.1", "localhost", "::1"):
        if local_host not in no_proxy:
            no_proxy.append(local_host)
    environment["NO_PROXY"] = ",".join(no_proxy)
    environment["no_proxy"] = environment["NO_PROXY"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command,
                cwd=VALIDATOR_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s or None,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"task process timed out after {timeout_s}s"}
    if not result_path.is_file():
        return {"status": "error", "error": f"task process exited {completed.returncode} without result JSON"}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if completed.returncode and result.get("status") == "ok":
        result["status"] = "error"
        result["error"] = f"task process exited {completed.returncode}"
    return result


def _summary(results: dict[int, dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    episodes = sum(int(result.get("num_trials", 0)) for result in results.values())
    successes = sum(int(result.get("num_successes", 0)) for result in results.values())
    failed_tasks = sum(result.get("status") != "ok" for result in results.values())
    return {
        **metadata,
        "tasks": {str(task_id): result for task_id, result in sorted(results.items())},
        "total_tasks": len(results),
        "failed_tasks": failed_tasks,
        "total_episodes": episodes,
        "total_successes": successes,
        "overall_success_rate": successes / episodes if episodes else 0.0,
        "suites": {
            metadata["benchmark"]: {
                "tasks": len(results),
                "episodes": episodes,
                "successes": successes,
                "success_rate": successes / episodes if episodes else 0.0,
            }
        },
    }


def run(
    args: Any,
    checkpoint: pathlib.Path | None,
    gpus: list[int],
    axis_python: pathlib.Path,
    *,
    start_servers: Callable[..., list[Any]] | None = None,
    wait_for_servers: Callable[[list[Any], float], None] | None = None,
    stop_servers: Callable[[list[Any]], None] | None = None,
) -> int:
    if not axis_python.is_file():
        raise FileNotFoundError(f"AXIS runtime is missing: {axis_python}; run bash setup_axis.sh")
    manifest_path = _manifest_path(args.benchmark, getattr(args, "axis_manifest", None))
    manifest = getattr(args, "axis_loaded_manifest", None) or load_manifest(
        manifest_path, expected_name=None if args.benchmark == "axis" else args.benchmark
    )
    benchmark = manifest["name"]
    if not manifest.get("protocol_revision"):
        raise ValueError("AXIS manifests must declare a non-empty protocol_revision")
    if "replan_steps" in manifest["protocol"] and args.axis_replan_steps != manifest["protocol"]["replan_steps"]:
        raise ValueError("--axis-replan-steps differs from the frozen manifest protocol")
    gripper_mode = getattr(args, "axis_gripper_mode", "continuous")
    policy_samples = getattr(args, "axis_policy_samples", 1)
    sample_reduction = getattr(args, "axis_sample_reduction", "mean")
    if sample_reduction != manifest["protocol"].get("sample_reduction", "mean"):
        raise ValueError("--axis-sample-reduction differs from the frozen manifest protocol")
    if policy_samples != manifest["protocol"].get("policy_samples", 1):
        raise ValueError("--axis-policy-samples differs from the frozen manifest protocol")
    if policy_samples > 1 and (args.axis_policy_port or gripper_mode != "continuous"):
        raise ValueError("policy sample averaging requires a validator-managed server and continuous gripper decoding")
    if gripper_mode != manifest["protocol"].get("gripper_mode", "continuous"):
        raise ValueError("--axis-gripper-mode differs from the frozen manifest protocol")
    if gripper_mode != "continuous" and args.axis_policy_port:
        raise ValueError("non-default gripper decoding requires a validator-managed policy server")
    if manifest["runtime"].get("renderer_backend") != AXIS_RENDERER_BACKEND:
        raise ValueError(
            f"{benchmark} renderer mismatch: "
            f"manifest={manifest['runtime'].get('renderer_backend')!r}, runtime={AXIS_RENDERER_BACKEND!r}"
        )
    selected = _selected_task_ids(args, manifest)
    if args.num_trials is None:
        args.num_trials = int(manifest["protocol"]["default_trials_per_task"])
    if args.num_trials < 1:
        raise ValueError("--num-trials must be at least 1")
    if manifest["protocol"]["randomization"]:
        if getattr(args, "axis_randomization_manifest", None) is None:
            raise ValueError("custom randomized Axis manifests require explicit --axis-randomization-manifest")
        if getattr(args, "axis_randomization_seed", None) is None:
            args.axis_randomization_seed = int(manifest["protocol"]["randomization_seed_default"])
    elif (
        getattr(args, "axis_randomization_manifest", None) is not None
        or getattr(args, "axis_randomization_seed", None) is not None
    ):
        raise ValueError(f"{benchmark} is base-only and rejects Axis randomization arguments")
    if args.suites or args.tasks or args.init_seed is not None or args.init_states_root:
        raise ValueError("AXIS uses manifest task ids; --suites/--tasks and LIBERO init-state options do not apply")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = "environment-smoke" if checkpoint is None else pathlib.Path(str(args.model).rstrip("/")).name
    output_dir = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else VALIDATOR_ROOT / "eval_runs" / f"{stamp}_{benchmark}_{model_tag}"
    )
    logs_dir, results_dir = output_dir / "logs", output_dir / "results"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    selection = getattr(args, "axis_task_selection", None)
    if selection is not None:
        (output_dir / "task_selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[run_eval] AXIS random sample: {selection['sample_size']}/{selection['pool_task_count']} "
            f"tasks, seed={selection['sampling_seed']}"
        )
    if manifest_path.suffix in (".yaml", ".yml"):
        # Children read a fixed JSON snapshot, so editing the YAML during a run
        # cannot change the remaining tasks or the score's manifest identity.
        snapshot_root = manifest["task_snapshot_root"]
        (output_dir / snapshot_root).mkdir(parents=True, exist_ok=True)
        for task_id in task_specs(manifest):
            shutil.copyfile(
                manifest_path.parent / snapshot_root / f"{task_id}.json",
                output_dir / snapshot_root / f"{task_id}.json",
            )
        resolved_path = output_dir / "benchmark_manifest.json"
        resolved_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        args.axis_resolved_manifest = str(resolved_path)
    print(f"[run_eval] AXIS tasks: {selected}")
    print(f"[run_eval] output dir: {output_dir}")

    started = time.perf_counter()
    results: dict[int, dict[str, Any]] = {}
    servers: list[Any] = []
    progress_path = getattr(args, "progress_file", None)
    # Environment-only smoke checks must not advertise completed model trials.
    progress_path = pathlib.Path(progress_path).resolve() if progress_path and not args.dry_run else None
    with report_axis_progress(progress_path, benchmark, selected, args.num_trials) as record_progress:
        try:
            if args.dry_run:
                for index, task_id in enumerate(selected):
                    result_path = results_dir / f"axis_{task_id}.json"
                    command = _task_command(
                        args,
                        axis_python,
                        task_id,
                        result_path,
                        port=args.axis_policy_port or args.base_port,
                        dry_run=True,
                    )
                    results[task_id] = _run_task(
                        command,
                        logs_dir / f"axis_{task_id}.log",
                        result_path,
                        gpus[index % len(gpus)],
                        args.task_timeout,
                    )
                    print(f"[{benchmark}] task {task_id}: {results[task_id].get('status')}")
            else:
                if args.axis_policy_port:
                    ports = [int(args.axis_policy_port)] * len(gpus)
                else:
                    if checkpoint is None or start_servers is None or wait_for_servers is None or stop_servers is None:
                        raise RuntimeError("AXIS model serving callbacks/checkpoint are unavailable")
                    if args.model_family != "openpi":
                        raise ValueError(
                            f"{benchmark} currently connects to the OpenPI-compatible policy contract; "
                            "the LingBot model-side adapter is still external work"
                        )
                    servers = start_servers(
                        gpus,
                        args.base_port,
                        args.config,
                        checkpoint,
                        logs_dir,
                        args.mem_fraction,
                        server_impl=args.server_impl,
                        max_batch=args.max_batch,
                        model_family=args.model_family,
                        seed=args.seed,
                        benchmark=args.benchmark,
                        axis_gripper_mode=gripper_mode,
                        axis_policy_samples=policy_samples,
                        axis_sample_reduction=sample_reduction,
                    )
                    wait_for_servers(servers, args.server_timeout)
                    ports = [server.port for server in servers]

                # Asset preparation is sequential so common Franka files are fetched once.
                for index, task_id in enumerate(selected):
                    prepare_result = results_dir / f"axis_{task_id}.prepare.json"
                    command = _task_command(
                        args,
                        axis_python,
                        task_id,
                        prepare_result,
                        port=ports[index % len(ports)],
                        dry_run=False,
                        prepare_only=True,
                    )
                    prepared = _run_task(
                        command,
                        logs_dir / f"axis_{task_id}.prepare.log",
                        prepare_result,
                        gpus[index % len(gpus)],
                        args.task_timeout,
                    )
                    if prepared.get("status") != "ok":
                        results[task_id] = prepared
                        record_progress(task_id, prepared)

                slots = [
                    (gpu, port) for gpu, port in zip(gpus, ports) for _ in range(max(1, int(args.workers_per_gpu)))
                ]

                def evaluate(task_id: int, gpu: int, port: int) -> tuple[int, dict[str, Any]]:
                    result_path = results_dir / f"axis_{task_id}.json"
                    command = _task_command(
                        args,
                        axis_python,
                        task_id,
                        result_path,
                        port=port,
                        dry_run=False,
                    )
                    result = _run_task(
                        command,
                        logs_dir / f"axis_{task_id}.log",
                        result_path,
                        gpu,
                        args.task_timeout,
                    )
                    return task_id, result

                pending = [task_id for task_id in selected if task_id not in results]
                if pending:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(slots), len(pending))) as executor:
                        futures = {
                            executor.submit(evaluate, task_id, *slots[index % len(slots)]): task_id
                            for index, task_id in enumerate(pending)
                        }
                        for future in concurrent.futures.as_completed(futures):
                            task_id, result = future.result()
                            results[task_id] = result
                            record_progress(task_id, result)
                            print(f"[{benchmark}] task {task_id}: {result.get('status')}")
        finally:
            if servers and stop_servers is not None:
                stop_servers(servers)

    summary = _summary(
        results,
        {
            "benchmark": benchmark,
            "protocol_revision": manifest["protocol_revision"],
            "renderer_backend": AXIS_RENDERER_BACKEND,
            "manifest_canonical_sha256": canonical_json_sha256(manifest),
            "evaluator_source_git_commit": args.evaluator_source_git_commit,
            "dry_run": bool(args.dry_run),
            "model": None if checkpoint is None else str(args.model),
            "model_family": None if checkpoint is None else args.model_family,
            "backbone": None if checkpoint is None else args.backbone,
            "commit_id": None if checkpoint is None else args.commit_id,
            "checkpoint_provenance": _checkpoint_provenance(checkpoint),
            "discrete_state_input": None if checkpoint is None else AXIS_PI05_DISCRETE_STATE_INPUT,
            "policy_seed": None if checkpoint is None else args.seed,
            "replan_steps": args.axis_replan_steps,
            "gripper_mode": gripper_mode,
            "policy_samples": policy_samples,
            "sample_reduction": sample_reduction,
            "num_trials_per_task": 0 if args.dry_run else args.num_trials,
            "randomization": bool(manifest["protocol"]["randomization"]),
            "randomization_seed": (args.axis_randomization_seed if manifest["protocol"]["randomization"] else None),
            "gpus": gpus,
            "wall_time_s": round(time.perf_counter() - started, 2),
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
        },
    )
    summary_path = output_dir / "summary.json"
    if selection is not None:
        summary["task_selection"] = selection
        summary["evaluation_scope"] = (
            "sampled_tasks" if selection["sample_size"] < selection["pool_task_count"] else "full_pool"
        )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[run_eval] AXIS summary written to {summary_path}")
    return 2 if summary["failed_tasks"] else 0
