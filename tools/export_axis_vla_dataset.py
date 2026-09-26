#!/usr/bin/env python3
"""Render AXIS expert replays with the frozen evaluation runtime.

Source images are not reused. This tool replays native 9D expert targets in the
same MuJoCo environment used by AXIS v1.0 evaluation,
keeps only trajectories that satisfy the checker, and records MuJoCo camera0
observations before every target.  The output intentionally carries
``eligible_for_scoring=false`` because replayed demonstrations are training data,
not policy evaluation results.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pathlib
import sys

import numpy as np


VALIDATOR_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VALIDATOR_ROOT / "libero_eval"))

from axis_runtime import (  # noqa: E402
    DEFAULT_MANIFEST,
    AssetCache,
    AxisEnvironment,
    canonical_json_sha256,
    fetch_task_payload,
    load_manifest,
    task_specs,
)
from axis_vla import SCHEMA_VERSION, TRAINING_PURPOSE, save_artifact  # noqa: E402


def _episode_bounds(episode_ends: np.ndarray, episode: int) -> tuple[int, int]:
    if episode < 0 or episode >= len(episode_ends):
        raise ValueError(f"episode must be in [0,{len(episode_ends) - 1}], got {episode}")
    return (0 if episode == 0 else int(episode_ends[episode - 1]), int(episode_ends[episode]))


def _replay(environment: AxisEnvironment, actions: np.ndarray, *, capture: bool) -> dict[str, object]:
    environment.reset()
    images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    success = False
    checker: dict[str, object] = {}
    for action in actions:
        if capture:
            images.append(environment.render())
            states.append(environment.observation_state())
            targets.append(np.asarray(action, dtype=np.float32))
        environment.step(action)
        success, checker = environment.success()
        if success:
            break
    return {
        "success": success,
        "checker": checker,
        "images": images,
        "states": states,
        "actions": targets,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--task-id", type=int, default=501)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=DEFAULT_MANIFEST,
        help="frozen benchmark YAML or JSON (default: AXIS v1.0)",
    )
    parser.add_argument(
        "--episodes", default="auto", help="'auto' keeps every successful replay, or comma-separated ids"
    )
    parser.add_argument("--source-frequency-hz", type=float, default=30.0, help="source target rate (default: 30 Hz)")
    parser.add_argument("--stride", type=int, default=6, help="source frames per control (30 Hz to 5 Hz: 6)")
    parser.add_argument("--cache-root", type=pathlib.Path, default=VALIDATOR_ROOT / ".cache" / "axis")
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error("--stride must be positive")
    if not math.isfinite(args.source_frequency_hz) or args.source_frequency_hz <= 0:
        parser.error("--source-frequency-hz must be finite and positive")

    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    if manifest["protocol"].get("randomization") is not False:
        raise ValueError("expert replay export requires a frozen base-scene manifest")
    specs = task_specs(manifest)
    if args.task_id not in specs:
        raise ValueError(f"task {args.task_id} is not in {manifest['name']}")
    spec = specs[args.task_id]
    runtime = manifest["runtime"]
    renderer_backend = str(runtime["renderer_backend"])
    if os.environ.get("MUJOCO_GL") != renderer_backend:
        raise ValueError(
            f"{manifest['name']} export requires MUJOCO_GL={renderer_backend!r}, got {os.environ.get('MUJOCO_GL')!r}"
        )
    # The benchmark defines control timing; dataset timing is supplied separately.
    # Current manifests do not embed source-dataset sampling rates.
    runtime_frequency_hz = 1.0 / float(runtime["control_period_s"])
    expected_stride = args.source_frequency_hz / runtime_frequency_hz
    if not np.isclose(args.stride, expected_stride):
        parser.error(
            f"--stride {args.stride} does not match source/runtime frequency ratio {expected_stride:g}; "
            "set --source-frequency-hz and --stride to match the source data and benchmark control period"
        )

    import zarr

    dataset_path = args.dataset.expanduser().resolve()
    group = zarr.open_group(str(dataset_path), mode="r")
    try:
        source_states = group["data/state"]
        source_actions = group["data/action"]
        episode_ends = np.asarray(group["meta/episode_ends"])
    except KeyError as exc:
        raise ValueError("source Zarr requires data/state, data/action and meta/episode_ends") from exc
    if source_states.ndim != 2 or source_actions.ndim != 2:
        raise ValueError("AXIS source state/action arrays must be matrices")
    if source_states.shape != source_actions.shape or source_states.shape[1] != 9:
        raise ValueError(
            f"expected matching 9D source state/action arrays, got {source_states.shape}/{source_actions.shape}"
        )
    if (
        episode_ends.ndim != 1
        or not len(episode_ends)
        or not np.issubdtype(episode_ends.dtype, np.integer)
        or np.any(episode_ends <= 0)
        or np.any(np.diff(episode_ends.astype(np.int64)) <= 0)
        or int(episode_ends[-1]) != source_actions.shape[0]
    ):
        raise ValueError("meta/episode_ends must be increasing positive integers covering all source frames")
    if args.episodes == "auto":
        candidates = list(range(len(episode_ends)))
    else:
        try:
            candidates = [int(value) for value in args.episodes.split(",")]
        except ValueError:
            parser.error("--episodes must be 'auto' or unique comma-separated episode indices")
        if len(candidates) != len(set(candidates)):
            parser.error("--episodes must contain unique comma-separated episode indices")
        for episode in candidates:
            _episode_bounds(episode_ends, episode)

    cache_root = args.cache_root.expanduser().resolve()
    payload = fetch_task_payload(
        spec,
        api_base_url=runtime["task_api_base_url"],
        selection_contract=int(runtime["selection_contract"]),
        cache_root=cache_root,
        snapshot_root=manifest_path.parent / manifest.get("task_snapshot_root", f"{manifest_path.stem}-tasks"),
        refresh=False,
    )
    scene_path, _ = AssetCache(cache_root / "assets", runtime["asset_base_url"], workers=8).prepare_scene(
        args.task_id, payload["mjcf_xml"]
    )
    environment = AxisEnvironment(scene_path, payload, runtime)
    try:
        qualified: list[int] = []
        for episode in candidates:
            start, stop = _episode_bounds(episode_ends, episode)
            sampled = np.asarray(source_actions[start : stop : args.stride], dtype=np.float32)
            result = _replay(environment, sampled, capture=False)
            if result["success"]:
                qualified.append(episode)
            print(f"episode {episode}: {'PASS' if result['success'] else 'fail'}")
        if not qualified:
            raise RuntimeError(f"no source episode passed the exact {manifest['name']} checker; refusing to export")

        images: list[np.ndarray] = []
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        exported_ends: list[int] = []
        replay_details: list[dict[str, object]] = []
        for episode in qualified:
            start, stop = _episode_bounds(episode_ends, episode)
            sampled = np.asarray(source_actions[start : stop : args.stride], dtype=np.float32)
            result = _replay(environment, sampled, capture=True)
            if not result["success"]:
                raise RuntimeError(f"qualified episode {episode} was not deterministic on replay")
            images.extend(result["images"])
            states.extend(result["states"])
            actions.extend(result["actions"])
            exported_ends.append(len(images))
            replay_details.append({
                "source_episode": episode,
                "source_frame_range": [start, stop],
                "sampled_targets": len(sampled),
                "exported_targets_until_success": len(result["actions"]),
                "checker": result["checker"],
            })
    finally:
        environment.close()

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "purpose": TRAINING_PURPOSE,
        "eligible_for_scoring": False,
        "leakage_warning": "rendered from benchmark scenes; replay success is not a policy evaluation score",
        "benchmark": manifest["name"],
        "task_id": args.task_id,
        "instruction": spec["instruction"],
        "source_dataset": dataset_path.name,
        "source_frequency_hz": args.source_frequency_hz,
        "runtime_frequency_hz": runtime_frequency_hz,
        "stride": args.stride,
        "action_semantics": runtime["action_semantics"],
        "camera": runtime["camera"],
        "renderer_backend": renderer_backend,
        "image_size": runtime["image_size"],
        "manifest_sha256": canonical_json_sha256(manifest),
        "task_contract_hashes": {
            "mjcf_sha256": spec["mjcf_sha256"],
            "checker_sha256": spec["checker_sha256"],
            "initial_state_sha256": spec["initial_state_sha256"],
        },
        "successful_replays": len(qualified),
        "source_episodes": qualified,
        "training_examples": len(images),
        "replays": replay_details,
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    artifact = save_artifact(
        args.output.expanduser().resolve(),
        metadata=metadata,
        images=np.asarray(images, dtype=np.uint8),
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        episode_ends=np.asarray(exported_ends, dtype=np.int64),
    )
    print(json.dumps(artifact.metadata, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
