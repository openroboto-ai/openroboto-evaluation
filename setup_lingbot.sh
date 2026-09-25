#!/bin/bash
# Build the isolated LingBot-VLA 2.0 policy runtime used by LIBERO evaluation.
set -euo pipefail

VALIDATOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="$VALIDATOR_ROOT/third_party"
LINGBOT_DIR="${LINGBOT_VLA_V2_DIR:-$THIRD_PARTY/lingbot-vla-v2}"
QWEN3_DIR="${QWEN3_VL_DIR:-$THIRD_PARTY/Qwen3-VL-4B-Instruct}"

LINGBOT_REF=951475ae1b1d87553e7dc47c97b53a3d695c0d13
QWEN3_REF=ebb281ec70b05090aa6165b016eac8ec08e71b17

command -v git >/dev/null || { echo "git not found"; exit 1; }
command -v uv >/dev/null || { echo "uv not found — install from https://docs.astral.sh/uv/"; exit 1; }
mkdir -p "$THIRD_PARTY"

if [ ! -e "$LINGBOT_DIR" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Robbyant/lingbot-vla-v2 "$LINGBOT_DIR"
fi
if [ ! -d "$LINGBOT_DIR/.git" ]; then
    echo "$LINGBOT_DIR exists but is not a git checkout"
    exit 1
fi
git -C "$LINGBOT_DIR" fetch origin "$LINGBOT_REF"
git -C "$LINGBOT_DIR" checkout --quiet "$LINGBOT_REF"

echo "== 1/2 LingBot-VLA 2.0 inference env =="
[ -x "$LINGBOT_DIR/.venv/bin/python" ] || uv venv --python 3.12 "$LINGBOT_DIR/.venv"
VIRTUAL_ENV="$LINGBOT_DIR/.venv" uv pip install -r "$LINGBOT_DIR/requirements.txt"
VIRTUAL_ENV="$LINGBOT_DIR/.venv" uv pip install setuptools wheel
VIRTUAL_ENV="$LINGBOT_DIR/.venv" uv pip install numpydantic==1.9.0 --no-deps
VIRTUAL_ENV="$LINGBOT_DIR/.venv" MAX_JOBS="${MAX_JOBS:-8}" \
    uv pip install flash-attn==2.8.3 --no-build-isolation
VIRTUAL_ENV="$LINGBOT_DIR/.venv" uv pip install --no-deps \
    "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
VIRTUAL_ENV="$LINGBOT_DIR/.venv" uv pip install -e "$LINGBOT_DIR" --no-deps

echo "== 2/2 pinned Qwen3-VL processor files =="
"$LINGBOT_DIR/.venv/bin/python" - "$QWEN3_DIR" "$QWEN3_REF" <<'PY'
import pathlib
import sys

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-VL-4B-Instruct",
    revision=sys.argv[2],
    local_dir=pathlib.Path(sys.argv[1]),
    ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.onnx"],
)
PY

echo "== LingBot LIBERO runtime ready =="
