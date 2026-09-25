#!/usr/bin/env python3
"""Append one external selector batch to a frozen validator benchmark.

The selector owns sampling; validator owns executable definitions. No network,
queue updates or deployment take place. See the bilingual repository README.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import pathlib
import re
import shutil
import sys
import tempfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_runtime import AXIS_V1_CONFIG_PATH, canonical_json_sha256, load_manifest, task_specs, verify_task_payload
from tools.export_axis_task_docs import export_docs
from tools.verify_axis_release import verify_release

BRIDGE_VERSION = "axis-selector-validator-v2"


def load_selector(directory: pathlib.Path):
    """Use an explicit local checkout, without copying its sampling algorithm."""
    path = directory.resolve() / "selector.py"
    spec = importlib.util.spec_from_file_location("external_axis_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_root(path: pathlib.Path, manifest: dict) -> pathlib.Path:
    return path.parent / manifest.get("task_snapshot_root", f"{path.stem}-tasks")


def next_name(previous: str) -> str:
    match = re.fullmatch(r"(axis[-_]v[0-9]+)\.([0-9]+)", previous)
    if match is None:
        raise ValueError("cannot increment this benchmark name; supply --name axis_vN.N")
    return f"{match[1]}.{int(match[2]) + 1}"


def read_receipt(path: pathlib.Path, manifest: dict, selector_identity: dict, runtime_hash: str) -> dict | None:
    if "selector_sync" not in manifest:
        return None
    if manifest["selector_sync"] != {"bridge": BRIDGE_VERSION, "receipt": "selection.json"}:
        raise ValueError("unsupported selector bridge receipt")
    receipt = json.loads((path.parent / "selection.json").read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("selector receipt must be an object")
    if (
        receipt.get("bridge") != BRIDGE_VERSION
        or receipt.get("selector") != selector_identity
        or receipt.get("runtime_pool_sha256") != runtime_hash
    ):
        raise ValueError("selector code, task pool or runtime pool changed; keep this sequence pinned")
    definition_path = path.with_suffix(".json") if path.suffix in (".yaml", ".yml") else path
    if receipt.get("verification") != verify_release(definition_path, snapshot_root(path, manifest)):
        raise ValueError("previous benchmark or snapshots changed since selection")
    if "configuration_verification" in receipt or path.suffix in (".yaml", ".yml"):
        config_path = path if path.suffix in (".yaml", ".yml") else path.with_suffix(".yaml")
        if receipt.get("configuration_verification") != verify_release(config_path, snapshot_root(path, manifest)):
            raise ValueError("previous YAML configuration changed since selection")
    if receipt.get("selected_task_ids") != list(task_specs(manifest)):
        raise ValueError("previous receipt task history differs from its benchmark")
    request = receipt.get("selector_request", {})
    if (
        request.get("selected_ids") != receipt.get("previous_task_ids")
        or request.get("current_count") != len(receipt.get("previous_task_ids", []))
        or receipt.get("previous_task_ids", []) + receipt.get("added_task_ids", []) != receipt["selected_task_ids"]
    ):
        raise ValueError("previous receipt contains inconsistent history")
    return receipt


def write_json(path: pathlib.Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sync(
    *,
    selector_root: pathlib.Path,
    runtime_pool: pathlib.Path,
    previous: pathlib.Path,
    output: pathlib.Path,
    seed: int | str,
    name: str | None = None,
    selection: pathlib.Path | None = None,
) -> dict:
    previous, runtime_pool, output = previous.resolve(), runtime_pool.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; frozen rounds are never overwritten: {output}")
    baseline, source = load_manifest(previous), load_manifest(runtime_pool)
    old_specs, source_specs = task_specs(baseline), task_specs(source)
    if len(old_specs) < 30 or (len(old_specs) - 30) % 10:
        raise ValueError("previous benchmark must contain 30 + 10 * N tasks")
    name = name or next_name(baseline["name"])
    if re.fullmatch(r"axis[-_]v[0-9]+(?:\.[0-9]+)?", name) is None or name == baseline["name"]:
        raise ValueError("name must be a new version, e.g. axis_v1.1")
    if baseline["protocol"].get("randomization") is not False:
        raise ValueError("this bridge requires a base-scene benchmark")
    if any(baseline[key] != source[key] for key in ("runtime", "protocol")):
        raise ValueError("runtime pool must use the same runtime and evaluation protocol as the previous benchmark")
    policy_seed = baseline.get("policy_seed")
    if type(policy_seed) is not int or not 0 <= policy_seed < 2**32:
        raise ValueError("previous benchmark must pin a uint32 policy_seed")

    selector, code_hash = load_selector(selector_root)
    pool = selector.load_pool()
    public_specs = selector.validate_pool(pool)
    missing = sorted(public_specs.keys() - source_specs.keys())
    if missing:
        raise ValueError(
            f"runtime pool lacks public selector tasks: {missing}; do not silently shrink the selector pool"
        )
    identity = {
        "algorithm": selector.ALGORITHM,
        "code_sha256": code_hash,
        "pool_sha256": selector.pool_fingerprint(pool),
        "pool_task_count": len(public_specs),
    }
    runtime_hash = canonical_json_sha256(source)
    prior_receipt = read_receipt(previous, baseline, identity, runtime_hash)
    if prior_receipt is not None:
        replay = selector.draw(**prior_receipt["selector_request"], pool=pool)
        if replay != prior_receipt["added_task_ids"]:
            raise ValueError("previous selector draw does not replay")

    # Use the same complete history as the standalone, published selector.
    history = list(old_specs)
    request = {"seed": "0x" + selector.normalize_seed(seed), "current_count": len(history), "selected_ids": history}
    expected = selector.draw(**request, pool=pool)
    added = json.loads(selection.read_text(encoding="utf-8")) if selection is not None else expected
    if (
        not isinstance(added, list)
        or len(added) != 10
        or any(type(task_id) is not int for task_id in added)
        or len(set(added)) != 10
        or not set(added) <= public_specs.keys()
        or set(added) & old_specs.keys()
    ):
        raise ValueError("selector must return exactly 10 distinct, unseen public task IDs")
    if added != expected:
        raise ValueError("selection does not match selector replay for the supplied seed and complete history")
    if len({public_specs[task_id]["task_group"] for task_id in added}) != 10:
        raise ValueError("selector batch must contain 10 distinct task groups")

    selected = list(old_specs) + added
    specs = {**old_specs, **{task_id: source_specs[task_id] for task_id in added}}
    # Retain old definitions and exact payload bytes, even if a newer pool has
    # a different definition under the same ID. No API fallback or redraw.
    payloads = {}
    for task_id in selected:
        path, manifest = (previous, baseline) if task_id in old_specs else (runtime_pool, source)
        raw = (snapshot_root(path, manifest) / f"{task_id}.json").read_bytes()
        verify_task_payload(json.loads(raw), specs[task_id])
        payloads[task_id] = raw
    manifest = {
        "schema_version": 1,
        "name": name,
        "status": "runtime-ready",
        "description": "Cumulative benchmark: preserved previous tasks plus 10 public AXIS selector tasks.",
        "protocol_revision": f"{name}_{len(selected)}tasks_selector_v1",
        "policy_seed": policy_seed,
        "protocol": copy.deepcopy(baseline["protocol"]),
        "runtime": copy.deepcopy(baseline["runtime"]),
        "tasks": [
            {**copy.deepcopy(specs[task_id]), "name_zh": specs[task_id].get("name_zh", specs[task_id]["instruction"])}
            for task_id in selected
        ],
        "task_snapshot_root": f"{name}-tasks",
        "selector_sync": {"bridge": BRIDGE_VERSION, "receipt": "selection.json"},
    }
    # Keep the same YAML schema as axis_v1.0. JSON remains the hash-pinned
    # definition store; YAML is the user-facing configuration for each round.
    config = {
        "schema_version": 1,
        "name": name,
        "source_manifest": f"{name}.json",
        "source_manifest_sha256": canonical_json_sha256(manifest),
        "protocol_revision": manifest["protocol_revision"],
        "policy_seed": policy_seed,
        "tasks": [{key: task[key] for key in ("task_id", "name_zh", "instruction")} for task in manifest["tasks"]],
    }
    receipt = {
        "bridge": BRIDGE_VERSION,
        "benchmark": name,
        "previous_benchmark": baseline["name"],
        "previous_manifest_sha256": canonical_json_sha256(baseline),
        "previous_receipt_sha256": canonical_json_sha256(prior_receipt) if prior_receipt else None,
        "runtime_pool_sha256": runtime_hash,
        "selector": identity,
        "selector_request": request,
        "previous_task_ids": list(old_specs),
        "previous_public_task_ids": sorted(old_specs.keys() & public_specs.keys()),
        "retained_legacy_task_ids": sorted(old_specs.keys() - public_specs.keys()),
        "added_task_ids": added,
        "selected_task_ids": selected,
        "validation_scope": "frozen definitions and selector replay; no model evaluation or deployment",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    claimed = False
    try:
        target = staging / f"{name}.json"
        payload_dir = staging / manifest["task_snapshot_root"]
        payload_dir.mkdir()
        for task_id, raw in payloads.items():
            (payload_dir / f"{task_id}.json").write_bytes(raw)
        write_json(target, manifest)
        config_path = staging / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        receipt["verification"] = verify_release(target, payload_dir)
        receipt["configuration_verification"] = verify_release(config_path, payload_dir)
        write_json(staging / "selection.json", receipt)
        export_docs(manifest_path=config_path, selector_root=selector_root, output=staging / "task-docs")
        # Claim the final name exclusively, then publish the complete bundle.
        output.mkdir(exist_ok=False)
        claimed = True
        staging.replace(output)
        claimed = False
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if claimed:
            output.rmdir()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-root", required=True, type=pathlib.Path, help="Local axis-task-selector checkout")
    parser.add_argument(
        "--runtime-pool", required=True, type=pathlib.Path, help="Frozen runtime manifest with snapshots"
    )
    parser.add_argument(
        "--previous", type=pathlib.Path, default=AXIS_V1_CONFIG_PATH, help="Previous YAML or JSON benchmark"
    )
    parser.add_argument(
        "--output", required=True, type=pathlib.Path, help="New directory for the complete next-round bundle"
    )
    parser.add_argument("--seed", required=True, help="Unsigned 256-bit decimal integer or 0x hexadecimal string")
    parser.add_argument(
        "--selection",
        type=pathlib.Path,
        help="JSON output from selector.py; verified against seed and complete history",
    )
    parser.add_argument(
        "--name", help="Defaults to incrementing the previous minor version, e.g. axis_v1.0 to axis_v1.1"
    )
    args = parser.parse_args()
    try:
        seed = args.seed if args.seed.startswith("0x") else int(args.seed)
        receipt = sync(
            selector_root=args.selector_root,
            runtime_pool=args.runtime_pool,
            previous=args.previous,
            output=args.output,
            seed=seed,
            name=args.name,
            selection=args.selection,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ImportError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "benchmark": receipt["benchmark"],
                "manifest": str(args.output.resolve() / f"{receipt['benchmark']}.yaml"),
                "tasks": len(receipt["selected_task_ids"]),
                "added_task_ids": receipt["added_task_ids"],
                "receipt": str(args.output.resolve() / "selection.json"),
                "task_docs": str(args.output.resolve() / "task-docs"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
