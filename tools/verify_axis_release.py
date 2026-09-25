#!/usr/bin/env python3
"""Verify the frozen AXIS manifest and task snapshot bundle without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_runtime import DEFAULT_MANIFEST, canonical_json_sha256, load_manifest, task_specs, verify_task_payload  # noqa: E402


def verify_release(manifest_path: pathlib.Path, snapshot_root: pathlib.Path) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    if manifest["protocol"].get("randomization") is not False:
        raise ValueError("base snapshot verification rejects randomized manifests")
    specs = task_specs(manifest)
    if not specs or len(specs) != len(manifest["tasks"]):
        raise ValueError("AXIS release must contain a nonempty unique task set")
    actual_files = sorted(path.name for path in snapshot_root.glob("*.json"))
    expected_files = sorted(f"{task_id}.json" for task_id in specs)
    if actual_files != expected_files:
        raise ValueError(f"AXIS task snapshot set mismatch: got {actual_files}, expected {expected_files}")

    snapshots: dict[str, dict[str, str]] = {}
    for task_id, spec in sorted(specs.items()):
        path = snapshot_root / f"{task_id}.json"
        raw = path.read_bytes()
        verified = verify_task_payload(json.loads(raw), spec)
        snapshots[str(task_id)] = {
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "payload_canonical_sha256": canonical_json_sha256(verified),
        }
    return {
        "benchmark": manifest["name"],
        "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "manifest_canonical_sha256": canonical_json_sha256(manifest),
        "snapshot_bundle_canonical_sha256": canonical_json_sha256(snapshots),
        "tasks": snapshots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument("--snapshot-root", type=pathlib.Path, default=None)
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    snapshot_root = (
        args.snapshot_root.expanduser().resolve()
        if args.snapshot_root is not None
        else manifest_path.with_name(f"{manifest_path.stem}-tasks")
    )
    print(json.dumps(verify_release(manifest_path, snapshot_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
