#!/usr/bin/env python3
"""Read-only audit of V6 scene preparation for a frozen task manifest.

This checks XML geometry/camera prerequisites, not task release provenance,
physics solvability, or a policy rollout. Output is explicitly an audit.
"""

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))
from axis_scene import CAMERA_CONFIG_SHA256, prepare_scene, resolve_profile
from axis_runtime import load_manifest, verify_task_payload


def audit(manifest_path: Path, assets: Path) -> dict:
    manifest = load_manifest(manifest_path)
    tasks = manifest_path.parent / manifest.get("task_snapshot_root", manifest_path.stem + "-tasks")
    rows = []
    # A scene recipe is an audit input, not a claim about any task's release seed.
    visual = dict(
        mode="official_franka_v6",
        camera_config_sha256=CAMERA_CONFIG_SHA256,
        global_seed=42,
        attempt_id=0,
        variant_id=0,
        replica_id=0,
    )
    assets.joinpath("scenes").mkdir(parents=True, exist_ok=True)
    for spec in manifest["tasks"]:
        tid = spec["task_id"]
        payload = verify_task_payload(json.loads((tasks / f"{tid}.json").read_bytes()), spec)
        row = {"task_id": tid, "instruction": spec["instruction"], "xml_preparation_passed": False}
        try:
            config, _ = resolve_profile(visual, tid, payload["name"])
            with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=assets / "scenes") as file:
                file.write(payload["mjcf_xml"])
                file.flush()
                _, metadata = prepare_scene(Path(file.name), config)
            row.update(
                xml_preparation_passed=True,
                table_center_xy=metadata["table_center_xy"],
                table_full_size=metadata["table_full_size"],
                fixture_count=metadata["fixture_count"],
            )
        except (ValueError, FileNotFoundError) as exc:
            row["error"] = str(exc)
        rows.append(row)
    return {
        "benchmark": manifest["name"],
        "task_count": len(rows),
        "xml_preparation_passed": sum(row["xml_preparation_passed"] for row in rows),
        "scope": "XML preparation only; no compilation, physical rollout or release binding certification",
        "audit_visual_inputs": visual,
        "tasks": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.manifest.resolve(), args.asset_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "tasks"}, ensure_ascii=False))
