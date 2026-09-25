#!/usr/bin/env python3
"""Download a pinned Hugging Face subtree from a local Git manifest.

This is a narrow recovery tool for large repos whose API-wide recursive listing
is unavailable.  The Git checkout supplies only path names; every file is still
resolved and downloaded from the requested immutable Hugging Face revision.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import subprocess
import threading

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars

from download import COMMIT_HASH_RE


def manifest_paths(
    manifest_repo: pathlib.Path,
    revision: str,
    prefix: str,
    exclude_prefixes: tuple[str, ...] = (),
) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(manifest_repo), "ls-tree", "-r", "--name-only", revision, "--", prefix],
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [line for line in result.stdout.splitlines() if line]
    return [path for path in paths if not any(path.startswith(item) for item in exclude_prefixes)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--exclude-prefix", action="append", default=[])
    parser.add_argument("--manifest-repo", type=pathlib.Path, required=True)
    parser.add_argument("--local-dir", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--no-token",
        action="store_true",
        help="Do not send the locally configured Hugging Face token (required for third-party mirrors)",
    )
    args = parser.parse_args()
    if not COMMIT_HASH_RE.fullmatch(args.revision):
        parser.error("--revision must be a full 40-character lowercase commit hash")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    manifest = manifest_paths(args.manifest_repo.resolve(), args.revision, args.prefix, tuple(args.exclude_prefix))
    if not manifest:
        raise SystemExit(f"no files found under {args.prefix!r} at {args.revision}")
    args.local_dir.mkdir(parents=True, exist_ok=True)
    paths = [path for path in manifest if not (args.local_dir / path).is_file()]
    already_present = len(manifest) - len(paths)
    if not paths:
        print(f"[download_hf_paths] all {len(manifest)} files are already present")
        return
    disable_progress_bars()
    done = 0
    progress_lock = threading.Lock()

    def download(path: str) -> None:
        nonlocal done
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            filename=path,
            local_dir=args.local_dir,
            endpoint=args.endpoint,
            token=False if args.no_token else None,
        )
        with progress_lock:
            done += 1
            if done == len(paths) or done % 100 == 0:
                print(f"[download_hf_paths] {done}/{len(paths)} files", flush=True)

    print(
        f"[download_hf_paths] downloading {len(paths)} files under {args.prefix!r} "
        f"from {args.repo_id}@{args.revision[:12]}",
        flush=True,
    )
    if already_present:
        print(f"[download_hf_paths] resuming after {already_present} completed files", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, path) for path in paths]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print(f"[download_hf_paths] complete: {args.local_dir.resolve()}")


if __name__ == "__main__":
    main()
