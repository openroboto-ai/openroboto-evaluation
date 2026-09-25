#!/usr/bin/env python3
"""Export concise bilingual task lists from a verified AXIS benchmark.

Verification stays internal. Published documents contain task IDs, names and
official Hub links, without runtime settings or internal training results.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_runtime import AXIS_V1_NAME, load_manifest
from tools.verify_axis_release import verify_release

HUB = "https://hub.axisrobotics.ai/explorer-1/task?id={}"
GUIDE = "https://docs.axisrobotics.ai/contributor-guide/getting-started"


def cell(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace("|", "&#124;").replace("\n", " ")


def normalized_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def export_docs(
    *, manifest_path: pathlib.Path, selector_root: pathlib.Path, output: pathlib.Path, status: str = "preview"
) -> None:
    if status not in {"baseline", "preview"}:
        raise ValueError("status must be baseline or preview")
    manifest = load_manifest(manifest_path)
    tasks = manifest["tasks"]
    snapshots = manifest_path.parent / manifest.get("task_snapshot_root", f"{manifest_path.stem}-tasks")
    verification = verify_release(manifest_path, snapshots)
    if status == "baseline" and (manifest["name"] != AXIS_V1_NAME or len(tasks) != 30):
        raise ValueError("baseline status requires the existing 30-task configuration")
    if "selector_sync" in manifest:
        receipt = json.loads((manifest_path.parent / "selection.json").read_text(encoding="utf-8"))
        key = "configuration_verification" if manifest_path.suffix in {".yaml", ".yml"} else "verification"
        if receipt.get(key) != verification or receipt.get("selected_task_ids") != [t["task_id"] for t in tasks]:
            raise ValueError("selection receipt does not match the benchmark being documented")
    hub_path = selector_root / "data/hub_task_links.json"
    hub = json.loads(hub_path.read_text(encoding="utf-8")) if hub_path.exists() else {"tasks": []}
    links = {row["task_id"]: row for row in hub["tasks"]}
    documents = {
        filename: render(manifest, links, status, zh) for filename, zh in (("README.md", False), ("readme_zh.md", True))
    }
    output.mkdir(parents=True, exist_ok=False)
    for filename, contents in documents.items():
        (output / filename).write_text(contents, encoding="utf-8")


def render(manifest: dict, links: dict, status: str, zh: bool) -> str:
    def tr(en: str, cn: str) -> str:
        return cn if zh else en

    tasks = manifest["tasks"]
    lines = [
        f"# {manifest['name']}",
        "[English](README.md) | [简体中文](readme_zh.md)",
        tr(
            f"This task set contains **{len(tasks)} tasks**, using official AXIS task IDs.",
            f"本任务集包含 **{len(tasks)} 个任务**，沿用 AXIS 官方任务编号。",
        ),
    ]
    if status == "preview":
        lines.append(tr("**PREVIEW — not published.**", "**预览——尚未公布。**"))
    lines.append(
        tr(
            f"Open **Hub** to view a task. For data collection, follow the [official contributor guide]({GUIDE}); "
            "an AXIS account and an available task are required.",
            f"点击 **Hub** 查看任务详情。采集数据请参照[官方贡献者指南]({GUIDE})，需登录 AXIS 且任务当前开放。",
        )
    )
    rows = [tr("| ID | Task | AXIS Hub |", "| 编号 | 任务 | AXIS Hub |"), "|---|---|---|"]
    for task in tasks:
        tid = task["task_id"]
        link = links.get(tid, {})
        name = task.get("name_zh", task["instruction"]) if zh else task["instruction"]
        if zh and name != task["instruction"]:
            name += f" / {task['instruction']}"
        matched = link.get("status") == "verified" and normalized_title(link.get("name", "")) == normalized_title(
            task["instruction"]
        )
        note = "" if matched else tr(" · not verified", " · 未核验")
        if link.get("status") == "title-mismatch" or (link.get("status") == "verified" and not matched):
            note = tr(" · Hub title: ", " · Hub 名称：") + cell(link["name"])
        rows.append(f"| {tid} | {cell(name)} | [Hub]({HUB.format(tid)}){note} |")
    return "\n\n".join([*lines, "\n".join(rows)]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--selector-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path, help="New documentation directory")
    parser.add_argument("--status", choices=("baseline", "preview"), default="preview")
    args = parser.parse_args()
    try:
        export_docs(
            manifest_path=args.manifest, selector_root=args.selector_root, output=args.output, status=args.status
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
