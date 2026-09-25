#!/usr/bin/env python3
"""Freeze hash-verified AXIS task payloads into release snapshots."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_runtime import DEFAULT_MANIFEST, load_manifest, task_specs, verify_task_payload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--source-root",
        type=pathlib.Path,
        default=ROOT / ".cache" / "axis" / "tasks",
        help="Directory containing previously downloaded task JSON files",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    if manifest["protocol"].get("randomization") is not False:
        parser.error("base snapshot freezing rejects randomized manifests; freeze their versioned variants instead")
    specs = task_specs(manifest)
    if not specs or len(specs) != len(manifest["tasks"]):
        parser.error("AXIS release must contain a nonempty unique task set")
    source_root = args.source_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else manifest_path.with_name(f"{manifest_path.stem}-tasks")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[int] = []
    for task_id, spec in sorted(specs.items()):
        source = source_root / f"{task_id}.json"
        if not source.is_file():
            parser.error(f"source task payload is missing: {source}")
        verified = verify_task_payload(json.loads(source.read_text(encoding="utf-8")), spec)
        destination = output_root / f"{task_id}.json"
        rendered = json.dumps(verified, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if destination.exists() and not args.overwrite:
            if destination.read_text(encoding="utf-8") != rendered:
                parser.error(f"snapshot exists with different content: {destination}; pass --overwrite explicitly")
            continue
        destination.write_text(rendered, encoding="utf-8")
        written.append(task_id)
    print(json.dumps({"output_root": str(output_root), "tasks": sorted(specs), "written": written}, indent=2))


if __name__ == "__main__":
    main()
