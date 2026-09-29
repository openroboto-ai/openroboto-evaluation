"""Evaluate one frozen Axis task against an OpenPI-compatible policy server."""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Any

from axis_randomization import build_trial_plan, load_randomization_plan
from axis_runtime import (
    DEFAULT_MANIFEST,
    AssetCache,
    AxisEnvironment,
    load_manifest,
    resolve_task_payload,
    task_specs,
    task_runtime,
    task_trial_count,
)


def _policy_client(host: str, port: int) -> Any:
    from openpi_client import websocket_client_policy

    return websocket_client_policy.WebsocketClientPolicy(host, port)


def _resize_image(image: Any, size: int) -> Any:
    from openpi_client import image_tools

    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, size, size))


def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class TrialPayload:
    data: dict[str, Any]
    source: str
    canonical_sha256: str
    randomization: dict[str, Any]


def _resolve_trial_payloads(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    spec: dict[str, Any],
    cache_root: pathlib.Path,
) -> list[TrialPayload]:
    if isinstance(args.num_trials, bool) or not isinstance(args.num_trials, int) or args.num_trials < 1:
        raise ValueError("AXIS task evaluation requires at least one trial")
    protocol = manifest["protocol"]
    randomized = bool(protocol["randomization"])
    randomization_manifest = getattr(args, "randomization_manifest", None)
    randomization_seed = getattr(args, "randomization_seed", None)
    if not randomized:
        if randomization_manifest is not None or randomization_seed is not None:
            raise ValueError(f"{manifest['name']} is base-only and rejects randomization arguments")
        resolved = resolve_task_payload(
            spec,
            api_base_url=args.task_api_base_url or manifest["runtime"]["task_api_base_url"],
            selection_contract=int(manifest["runtime"]["selection_contract"]),
            cache_root=cache_root,
            snapshot_root=pathlib.Path(args.manifest).resolve().parent
            / manifest.get("task_snapshot_root", f"{pathlib.Path(args.manifest).stem}-tasks"),
            refresh=args.refresh_task,
        )
        if "official_randomization" in resolved.data:
            raise ValueError("base-only AXIS manifests cannot run randomized payloads")
        return [
            TrialPayload(
                data=resolved.data,
                source=resolved.source,
                canonical_sha256=resolved.canonical_sha256,
                randomization={
                    "enabled": False,
                    "trial": trial,
                    "variant_id": "base",
                    "variant_index": 0,
                    "variant_count": 1,
                    "payload_canonical_sha256": resolved.canonical_sha256,
                },
            )
            for trial in range(args.num_trials)
        ]

    protocol_revision = manifest.get("protocol_revision") or protocol.get("revision")
    if not isinstance(protocol_revision, str) or not protocol_revision:
        raise ValueError("randomized AXIS manifest must freeze protocol_revision")
    if randomization_manifest is None or randomization_seed is None:
        raise ValueError("randomized AXIS evaluation requires --randomization-manifest and --randomization-seed")
    plan = load_randomization_plan(
        pathlib.Path(randomization_manifest),
        expected_benchmark=manifest["name"],
        expected_protocol_revision=protocol_revision,
        benchmark_task_specs=task_specs(manifest),
    )
    return [
        TrialPayload(
            data=item.payload,
            source="frozen-randomization-snapshot",
            canonical_sha256=item.selection.spec.payload_canonical_sha256,
            randomization=item.selection.provenance(),
        )
        for item in build_trial_plan(
            plan,
            task_id=int(spec["task_id"]),
            num_trials=args.num_trials,
            seed=randomization_seed,
        )
    ]


def _payload_metadata(trials: list[TrialPayload]) -> dict[str, Any]:
    sources = sorted({trial.source for trial in trials})
    payload_hashes = sorted({trial.canonical_sha256 for trial in trials})
    return {
        "task_payload_source": sources[0] if len(sources) == 1 else sources,
        "task_payload_sha256": payload_hashes[0] if len(payload_hashes) == 1 else None,
        "task_payload_sha256_set": payload_hashes,
        "trial_variants": [trial.randomization for trial in trials],
    }


def _smoke_environment(
    environment: AxisEnvironment, args: argparse.Namespace, runtime: dict[str, Any]
) -> dict[str, Any]:
    environment.reset()
    image = environment.render()
    initial_checker_passed, initial_checker = environment.success()
    physics_start = time.perf_counter()
    for _ in range(args.smoke_control_steps):
        environment.hold_current_pose()
        environment.mujoco.mj_step(
            environment.model,
            environment.data,
            nstep=environment.steps_per_control,
        )
    physics_s = time.perf_counter() - physics_start
    render_start = time.perf_counter()
    checksum = 0
    for _ in range(args.smoke_render_frames):
        checksum += int(environment.render()[0, 0, 0])
    render_s = time.perf_counter() - render_start
    return {
        "mujoco": {
            "nq": environment.model.nq,
            "nv": environment.model.nv,
            "nu": environment.model.nu,
            "timestep": float(environment.model.opt.timestep),
            "steps_per_control": environment.steps_per_control,
        },
        "smoke": {
            "image_shape": list(image.shape),
            "image_checksum": checksum,
            "physics_control_steps": args.smoke_control_steps,
            "physics_s": round(physics_s, 4),
            "physics_realtime_factor": round(
                args.smoke_control_steps * float(runtime["control_period_s"]) / physics_s,
                2,
            ),
            "render_frames": args.smoke_render_frames,
            "render_s": round(render_s, 4),
            "render_fps": round(args.smoke_render_frames / render_s, 2),
            "initial_checker_passed": initial_checker_passed,
            "initial_checker": initial_checker,
            **(
                {"official_randomization": environment.reset_randomization}
                if getattr(environment, "reset_randomization", None) is not None
                else {}
            ),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(pathlib.Path(args.manifest))
    benchmark = manifest["name"]
    specs = task_specs(manifest)
    if args.task_id not in specs:
        raise ValueError(f"task {args.task_id} is not part of {benchmark}; choose from {sorted(specs)}")
    spec = specs[args.task_id]
    runtime = task_runtime(manifest, spec)
    args.num_trials = task_trial_count(spec, args.num_trials)
    record_trials = getattr(args, "record_trials", 0)
    if record_trials < 0:
        raise ValueError("--record-trials must be non-negative")
    renderer_backend = str(runtime["renderer_backend"])
    if os.environ.get("MUJOCO_GL") != renderer_backend:
        raise ValueError(f"{benchmark} requires MUJOCO_GL={renderer_backend!r}, got {os.environ.get('MUJOCO_GL')!r}")
    cache_root = pathlib.Path(args.cache_root).expanduser().resolve()

    prepare_start = time.perf_counter()
    trials = _resolve_trial_payloads(args, manifest, spec, cache_root)
    payload_metadata = _payload_metadata(trials)
    asset_cache = AssetCache(
        cache_root / "assets",
        args.asset_base_url or runtime["asset_base_url"],
        workers=args.asset_fetch_workers,
    )
    unique_trials: dict[str, TrialPayload] = {}
    for trial in trials:
        unique_trials.setdefault(trial.canonical_sha256, trial)
    prepared: dict[str, tuple[pathlib.Path, dict[str, int]]] = {}
    protocol_randomized = bool(manifest["protocol"]["randomization"])
    randomized = protocol_randomized and spec.get("randomization_enabled", True)
    for payload_sha256, trial in unique_trials.items():
        if "official_randomization" in trial.data:
            from axis_perturbations import install_randomization_assets

            install_randomization_assets(trial.data["official_randomization"], asset_cache.root)
        prepared[payload_sha256] = asset_cache.prepare_scene(
            args.task_id,
            trial.data["mjcf_xml"],
            scene_key=payload_sha256 if protocol_randomized else None,
        )
    prepare_s = time.perf_counter() - prepare_start
    asset_counts_by_payload = {payload_sha256: counts for payload_sha256, (_, counts) in sorted(prepared.items())}
    asset_counts: dict[str, Any] = (
        next(iter(asset_counts_by_payload.values()))
        if len(asset_counts_by_payload) == 1
        else {"unique_variants": len(asset_counts_by_payload), "by_payload": asset_counts_by_payload}
    )
    task_name = trials[0].data["name"]

    if args.prepare_only:
        return {
            "status": "ok",
            "benchmark": benchmark,
            "task_id": args.task_id,
            "task_name": task_name,
            **payload_metadata,
            "randomization": randomized,
            "prepare_only": True,
            "asset_counts": asset_counts,
            "prepare_s": round(prepare_s, 4),
        }

    compile_start = time.perf_counter()
    environments: dict[str, AxisEnvironment] = {}
    try:
        for payload_sha256, trial in unique_trials.items():
            environments[payload_sha256] = AxisEnvironment(prepared[payload_sha256][0], trial.data, runtime)
        compile_s = time.perf_counter() - compile_start
        if args.dry_run:
            variant_smoke = {
                payload_sha256: _smoke_environment(environments[payload_sha256], args, runtime)
                for payload_sha256 in sorted(environments)
            }
            return {
                "status": "ok",
                "benchmark": benchmark,
                "task_id": args.task_id,
                "task_name": task_name,
                **payload_metadata,
                "renderer_backend": renderer_backend,
                "randomization": randomized,
                "dry_run": True,
                "asset_counts": asset_counts,
                "prepare_s": round(prepare_s, 4),
                "compile_s": round(compile_s, 4),
                **(next(iter(variant_smoke.values())) if len(variant_smoke) == 1 else {}),
                "variant_smoke": variant_smoke if len(variant_smoke) > 1 else None,
            }

        client = _policy_client(args.host, args.port)
        episodes: list[dict[str, Any]] = []
        successes = 0
        max_steps = int(args.max_control_steps or manifest["protocol"]["max_control_steps_per_trial"])
        for trial, trial_payload in enumerate(trials):
            episode_start = time.perf_counter()
            environment = environments[trial_payload.canonical_sha256]
            environment.reset()
            action_plan: collections.deque[Any] = collections.deque()
            inference_s = 0.0
            simulation_s = 0.0
            inference_calls = 0
            checker_detail: dict[str, Any] = {}
            error: str | None = None
            success = False
            steps = 0
            frames, trace = [], []
            recording = None
            try:
                if trial < record_trials:
                    frames.append(environment.render().copy())
                while steps < max_steps:
                    if not action_plan:
                        image = _resize_image(environment.render(), args.resize_size)
                        request = {
                            "observation/image": image,
                            "observation/state": environment.observation_state(),
                            "prompt": spec["instruction"],
                            "_axis_policy_task_id": args.task_id,
                            "_axis_policy_trial": trial,
                            "_axis_policy_call": inference_calls,
                            "_axis_variant_id": trial_payload.randomization["variant_id"],
                            "_axis_randomization_seed": trial_payload.randomization.get("seed"),
                        }
                        if runtime.get("wrist_camera") is not None:
                            request["observation/wrist_image"] = _resize_image(
                                environment.render(camera=runtime["wrist_camera"]), args.resize_size
                            )
                        mark = time.perf_counter()
                        response = client.infer(request)
                        inference_s += time.perf_counter() - mark
                        inference_calls += 1
                        actions = response.get("actions", response.get("action"))
                        if actions is None:
                            raise ValueError("policy response has neither 'actions' nor 'action'")
                        import numpy as np

                        chunk = np.asarray(actions, dtype=np.float64)
                        if chunk.ndim == 1:
                            chunk = chunk[None, :]
                        if chunk.ndim != 2 or chunk.shape[1] != len(runtime["observation_joint_order"]):
                            raise ValueError(f"policy action chunk must have shape [T,9], got {tuple(chunk.shape)}")
                        action_plan.extend(chunk[: args.replan_steps])
                    mark = time.perf_counter()
                    action = action_plan.popleft()
                    environment.step(action)
                    simulation_s += time.perf_counter() - mark
                    steps += 1
                    success, checker_detail = environment.success()
                    if trial < record_trials:
                        frames.append(environment.render().copy())
                        trace.append({
                            "step": steps,
                            "action": action.tolist(),
                            "state": environment.observation_state().tolist(),
                            "checker": checker_detail,
                        })
                    if success:
                        successes += 1
                        break
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"

            if frames:
                from axis_recording import save_recording

                recording = save_recording(
                    pathlib.Path(args.result_path).parent / "recordings",
                    task_id=args.task_id,
                    trial=trial,
                    frames=frames,
                    steps=trace,
                    control_period_s=float(runtime["control_period_s"]),
                    metadata={
                        "benchmark": benchmark,
                        "success": success,
                        "error": error,
                        "policy_seed": args.policy_seed,
                        "randomization": trial_payload.randomization,
                    },
                )

            episodes.append({
                "trial": trial,
                "randomization": trial_payload.randomization,
                **(
                    {"official_randomization": environment.reset_randomization}
                    if getattr(environment, "reset_randomization", None) is not None
                    else {}
                ),
                "success": success,
                "steps": steps,
                "duration_s": round(time.perf_counter() - episode_start, 4),
                "inference_calls": inference_calls,
                "timing": {
                    "inference_s": round(inference_s, 4),
                    "simulation_s": round(simulation_s, 4),
                },
                "checker": checker_detail,
                "error": error,
                **({"recording": recording} if recording is not None else {}),
            })

        return {
            "status": "ok" if all(episode["error"] is None for episode in episodes) else "error",
            "benchmark": benchmark,
            "task_id": args.task_id,
            "task_name": task_name,
            **payload_metadata,
            "renderer_backend": renderer_backend,
            "randomization": randomized,
            "policy_seed": args.policy_seed,
            "num_trials": len(episodes),
            "num_successes": successes,
            "success_rate": successes / len(episodes) if episodes else 0.0,
            "episodes": episodes,
            "prepare_s": round(prepare_s, 4),
            "compile_s": round(compile_s, 4),
            "asset_counts": asset_counts,
        }
    finally:
        for environment in environments.values():
            environment.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--task-api-base-url", default=None)
    parser.add_argument("--asset-base-url", default=None)
    parser.add_argument("--asset-fetch-workers", type=int, default=16)
    parser.add_argument("--refresh-task", action="store_true")
    parser.add_argument("--randomization-manifest", default=None)
    parser.add_argument("--randomization-seed", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-control-steps", type=int, default=None)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument(
        "--record-trials", type=int, default=0, help="Save GIF and per-step state/checker for first N trials"
    )
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--smoke-control-steps", type=int, default=100)
    parser.add_argument("--smoke-render-frames", type=int, default=100)
    parser.add_argument("--result-path", required=True)
    args = parser.parse_args()
    result_path = pathlib.Path(args.result_path)
    try:
        result = run(args)
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "error",
            "benchmark": pathlib.Path(args.manifest).stem,
            "task_id": args.task_id,
            "error": f"{type(exc).__name__}: {exc}",
        }
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
