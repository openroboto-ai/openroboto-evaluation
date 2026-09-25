#!/usr/bin/env python3
"""Preview AXIS task sampling without downloading a model or starting a GPU."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_runtime import AXIS_V1_CONFIG_PATH, load_manifest
from axis_sampling import sample_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=AXIS_V1_CONFIG_PATH)
    parser.add_argument("--count", type=int, required=True, help="Number of tasks to draw without replacement")
    parser.add_argument("--seed", type=int, help="Optional repeatable seed; omitted means fresh local randomness")
    parser.add_argument("--output", type=pathlib.Path, help="Optional new file for the sampling record")
    args = parser.parse_args()
    try:
        selection = sample_tasks(load_manifest(args.manifest), args.count, args.seed)
        serialized = json.dumps(selection, ensure_ascii=False, indent=2) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
