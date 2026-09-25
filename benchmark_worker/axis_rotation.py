"""Prepare immutable AXIS rounds from the backend's read-only rotation contract."""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import pathlib
import tempfile

from benchmark_worker.profiles import (
    AXIS_VERSION_PATTERN,
    block_axis_profile,
    get_profile,
    refresh_axis_profiles,
)
from libero_eval.axis_runtime import canonical_json_sha256, load_manifest, task_specs
from tools.sync_axis_benchmark import load_selector, next_name, read_receipt, sync


def default_rotation_directory(backend_url: str) -> pathlib.Path:
    identity = hashlib.sha256(backend_url.rstrip("/").encode()).hexdigest()[:16]
    return pathlib.Path(__file__).resolve().parents[1] / ".cache/axis/benchmarks" / identity


def normalize_rotation(value: dict, selector) -> dict:
    for field in ("benchmark", "previous_benchmark"):
        if not isinstance(value.get(field), str) or not AXIS_VERSION_PATTERN.fullmatch(value[field]):
            raise ValueError(f"rotation {field} must be an AXIS version, e.g. axis_v1.1")
    if value["benchmark"] != next_name(value["previous_benchmark"]):
        raise ValueError("rotation benchmark must be the next minor version of previous_benchmark")
    seed = value.get("seed")
    if not isinstance(seed, str) or not seed.startswith("0x"):
        raise ValueError("rotation seed must be a hexadecimal block hash beginning with 0x")
    seed = "0x" + selector.normalize_seed(seed)
    block = value.get("seed_block")
    if type(block) is not int or block < 0:
        raise ValueError("rotation seed_block must be a nonnegative integer")
    opens_at = value.get("opens_at")
    if not isinstance(opens_at, str):
        raise ValueError("rotation opens_at must be an ISO-8601 timestamp with a timezone")
    parsed = datetime.datetime.fromisoformat(opens_at.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("rotation opens_at must include a timezone")
    return {
        "benchmark": value["benchmark"],
        "previous_benchmark": value["previous_benchmark"],
        "seed": seed,
        "seed_block": block,
        "opens_at": parsed.astimezone(datetime.timezone.utc).isoformat(),
    }


class AxisRotation:
    """One backend namespace, pinned selector and runtime pool; no evaluation or API writes."""

    def __init__(
        self,
        *,
        directory: pathlib.Path,
        backend_url: str,
        selector_root: pathlib.Path,
        runtime_pool: pathlib.Path,
    ):
        self.directory = directory.resolve()
        self.selector_root = selector_root.resolve()
        self.runtime_pool = runtime_pool.resolve()
        self.selector, self.identity, self.runtime_hash = self._current_identity()
        self.context = {
            "backend_url": backend_url.rstrip("/"),
            "selector": self.identity,
            "runtime_pool_sha256": self.runtime_hash,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".prepare.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            context = self.directory / "rotation-context.json"
            if context.exists():
                if json.loads(context.read_text(encoding="utf-8")) != self.context:
                    raise ValueError(
                        "rotation directory belongs to a different backend or pinned selector/runtime pool"
                    )
            else:
                # Same-filesystem rename keeps another starting process from reading partial JSON.
                with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, delete=False) as stream:
                    temporary = pathlib.Path(stream.name)
                    json.dump(self.context, stream, indent=2, sort_keys=True)
                temporary.replace(context)

    def _current_identity(self):
        selector, code_hash = load_selector(self.selector_root)
        pool = selector.load_pool()
        tasks = selector.validate_pool(pool)
        source = load_manifest(self.runtime_pool)
        if not tasks.keys() <= task_specs(source).keys():
            raise ValueError("runtime pool lacks tasks from the pinned selector pool")
        identity = {
            "algorithm": selector.ALGORITHM,
            "code_sha256": code_hash,
            "pool_sha256": selector.pool_fingerprint(pool),
            "pool_task_count": len(tasks),
        }
        return selector, identity, canonical_json_sha256(source)

    def _verify_prepared(self, directory: pathlib.Path, request: dict, previous: pathlib.Path) -> None:
        path = directory / f"{request['benchmark']}.yaml"
        manifest = load_manifest(path, expected_name=request["benchmark"])
        receipt = read_receipt(path, manifest, self.identity, self.runtime_hash)
        if receipt is None:
            raise ValueError("rotation bundle is missing its selector receipt")
        previous_manifest = load_manifest(previous, expected_name=request["previous_benchmark"])
        if (
            receipt["previous_manifest_sha256"] != canonical_json_sha256(previous_manifest)
            or receipt["previous_benchmark"] != request["previous_benchmark"]
            or receipt["selector_request"]["seed"] != request["seed"]
            or receipt["previous_task_ids"] != list(task_specs(previous_manifest))
            or receipt["added_task_ids"]
            != self.selector.draw(**receipt["selector_request"], pool=self.selector.load_pool())
        ):
            raise ValueError("existing rotation does not match the requested seed, previous version or selector replay")
        recorded = json.loads((directory / "rotation.json").read_text(encoding="utf-8"))
        # Opening can be postponed by the backend; it never changes the frozen task set.
        if not isinstance(recorded, dict) or any(
            recorded.get(key) != request[key] for key in ("benchmark", "previous_benchmark", "seed", "seed_block")
        ):
            raise ValueError("existing rotation has conflicting block/seed metadata")

    def prepare(self, value: dict) -> pathlib.Path | None:
        name = value.get("benchmark")
        try:
            return self._prepare(value)
        except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
            if isinstance(name, str) and AXIS_VERSION_PATTERN.fullmatch(name):
                block_axis_profile(name, str(exc))
            raise

    def _prepare(self, value: dict) -> pathlib.Path | None:
        request = normalize_rotation(value, self.selector)
        name = request["benchmark"]
        with (self.directory / ".prepare.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return None  # Another local worker is preparing; do not claim the baseline yet.
            _, identity, runtime_hash = self._current_identity()
            if identity != self.identity or runtime_hash != self.runtime_hash:
                raise ValueError("selector or runtime pool changed while the worker was running")
            previous = get_profile(request["previous_benchmark"]).manifest_path
            if previous is None:
                raise ValueError("previous benchmark has no frozen AXIS configuration")
            target = self.directory / name
            if target.exists():
                self._verify_prepared(target, request, previous)
            else:
                with tempfile.TemporaryDirectory(prefix=".preparing-", dir=self.directory) as temp:
                    bundle = pathlib.Path(temp) / name
                    sync(
                        selector_root=self.selector_root,
                        runtime_pool=self.runtime_pool,
                        previous=previous,
                        output=bundle,
                        seed=request["seed"],
                        name=name,
                    )
                    (bundle / "rotation.json").write_text(
                        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    self._verify_prepared(bundle, request, previous)
                    bundle.rename(target)
            refresh_axis_profiles()
            block_axis_profile(name, None)
            get_profile(name)  # A conflicting or incomplete on-disk version must still fail closed.
            return target / f"{name}.yaml"
