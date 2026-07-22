"""Upload an openpi checkpoint directory to a Hugging Face model repo.

Token is taken from the HF_TOKEN environment variable. On this machine large
transfers can be killed mid-flight; upload_large_folder commits per-file and
resumes from where it left off, so simply re-run on failure.

Usage:
    HF_TOKEN=... python validator/tools/upload_hf.py \
        --src ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
        --repo user/pi05-libero-pytorch [--public]
"""

import argparse
import pathlib
import sys

from huggingface_hub import HfApi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="checkpoint directory to upload")
    ap.add_argument("--repo", required=True, help="target repo id, e.g. user/model-name")
    ap.add_argument("--public", action="store_true", help="create the repo as public (default: private)")
    ap.add_argument("--readme", default="", help="optional README.md content for the model card")
    args = ap.parse_args()

    src = pathlib.Path(args.src).expanduser()
    if not src.is_dir():
        sys.exit(f"[upload_hf] source directory not found: {src}")

    api = HfApi()
    user = api.whoami()["name"]
    print(f"[upload_hf] authenticated as {user}")

    api.create_repo(repo_id=args.repo, repo_type="model", private=not args.public, exist_ok=True)
    print(f"[upload_hf] repo ready: {args.repo} ({'public' if args.public else 'private'})")

    if args.readme:
        api.upload_file(
            path_or_fileobj=args.readme.encode(),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="model",
        )
        print("[upload_hf] README.md uploaded")

    api.upload_large_folder(repo_id=args.repo, repo_type="model", folder_path=str(src))
    print("[upload_hf] upload complete")

    files = api.list_repo_files(args.repo, repo_type="model")
    print(f"[upload_hf] repo files: {sorted(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
