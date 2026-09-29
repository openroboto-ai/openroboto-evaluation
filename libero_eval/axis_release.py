"""Install the current, hash-pinned AXIS bundle into a verified local cache."""

from __future__ import annotations

import fcntl
import hashlib
import json
import pathlib
import tarfile
import tempfile

import zstandard

AXIS_CURRENT_NAME = "axis_v2.0"
ROOT = pathlib.Path(__file__).resolve().parents[1]
RELEASE_DIRECTORY = ROOT / "configs" / "benchmarks"
CACHE_DIRECTORY = ROOT / ".cache" / "axis" / "releases"


def prepare_release(
    name: str = AXIS_CURRENT_NAME,
    *,
    release_directory: pathlib.Path = RELEASE_DIRECTORY,
    cache_directory: pathlib.Path = CACHE_DIRECTORY,
) -> pathlib.Path:
    """Verify archive and cached bytes before returning the frozen manifest.

    Existing corrupt caches fail rather than being silently repaired. A lock and
    atomic directory rename keep concurrent worker/CLI startups from seeing a
    partial extraction. Only the published archive determines cache contents.
    """
    if name not in (AXIS_CURRENT_NAME, "axis_v20260928.33"):
        raise ValueError(f"unsupported bundled AXIS release: {name}")
    receipt = json.loads((release_directory / f"{name}-release.json").read_bytes())
    archive = release_directory / f"{name}.tar.zst"
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != receipt["sha256"] or archive.stat().st_size != receipt["bytes"]:
        raise ValueError(f"AXIS archive checksum mismatch: {archive}")
    cache_directory.mkdir(parents=True, exist_ok=True)
    destination = cache_directory / name
    with (cache_directory / f".{name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_symlink():
            raise ValueError(f"AXIS cache must not be a symlink: {destination}")
        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=cache_directory) as temporary:
            target = destination if destination.exists() else pathlib.Path(temporary) / name
            target.mkdir(exist_ok=True)
            existing = destination.exists()
            seen = set()
            with archive.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as stream:
                with tarfile.open(fileobj=stream, mode="r|") as bundle:
                    for member in bundle:
                        relative = pathlib.PurePosixPath(member.name)
                        if relative.is_absolute() or ".." in relative.parts or "\\" in member.name:
                            raise ValueError(f"unsafe AXIS archive member: {member.name}")
                        path = target / relative
                        if path.is_symlink() or not path.resolve().is_relative_to(target.resolve()):
                            raise ValueError(f"AXIS cache path escapes its directory: {path}")
                        if member.isdir():
                            path.mkdir(parents=True, exist_ok=True)
                            continue
                        if not member.isfile() or relative.suffix != ".json" or str(relative) in seen:
                            raise ValueError(f"invalid AXIS archive member: {member.name}")
                        seen.add(str(relative))
                        raw = bundle.extractfile(member).read()
                        if existing:
                            if not path.is_file() or path.read_bytes() != raw:
                                raise ValueError(f"AXIS cache differs from the published archive: {path}")
                        else:
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(raw)
            if not {"benchmark.json", "randomization.json", "bindings.json", "validation.json"} <= seen:
                raise ValueError("AXIS bundle is missing required manifests or validation evidence")
            if {str(p.relative_to(target)) for p in target.rglob("*.json")} != seen:
                raise ValueError(f"AXIS cache contains unexpected JSON files: {target}")
            if not existing:
                target.rename(destination)
    return destination / "benchmark.json"
