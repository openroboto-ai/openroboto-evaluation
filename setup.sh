#!/bin/bash
# One-time setup for the standalone validator.
#
#   bash setup.sh [--with-checkpoint] [--with-openvla-oft]
#
# Builds, in order:
#   1. third_party checkouts   openpi (+ its LIBERO submodule), LIBERO-PRO and
#      LIBERO-plus. Skipped for a checkout that already exists — including
#      symlinks you created yourself, or locations overridden via
#      OPENPI_DIR / LIBERO_PRO_DIR / LIBERO_PLUS_DIR.
#      OPENVLA_OFT_DIR is used only when --with-openvla-oft is requested.
#   2. validator env           uv sync of this repo's pyproject.toml (orchestrator).
#   3. openpi server env       uv sync inside the openpi checkout (JAX/torch CUDA),
#                              plus openpi's transformers patch (PyTorch checkpoints).
#   4. LIBERO client env       Python 3.8 venv (MuJoCo/robosuite; LIBERO does not
#                              run on modern Python) inside openpi/examples/libero,
#                              plus wand/scikit-image for LIBERO-plus (needs the
#                              ImageMagick system library, see the check there).
#   5. demo checkpoint         pi05_libero (12.4 GB, GCS) — only with --with-checkpoint.
#
# The LIBERO-plus asset pack (6.4 GB, HF dataset Sylvest/LIBERO-plus) is NOT
# downloaded here — run_eval.py fetches and extracts it to ~/.cache/libero_plus
# on the first `--benchmark libero_plus` run.
#
# Requires: git, curl, and uv (https://docs.astral.sh/uv/).
set -e

VALIDATOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="$VALIDATOR_ROOT/third_party"
OPENPI_DIR="${OPENPI_DIR:-$THIRD_PARTY/openpi}"
LIBERO_PRO_DIR="${LIBERO_PRO_DIR:-$THIRD_PARTY/LIBERO-PRO}"
LIBERO_PLUS_DIR="${LIBERO_PLUS_DIR:-$THIRD_PARTY/LIBERO-plus}"
OPENVLA_OFT_DIR="${OPENVLA_OFT_DIR:-$THIRD_PARTY/openvla-oft}"

# Pinned upstream commits — the versions this pipeline was validated against.
# Bump deliberately (upstream changes can move uv.lock, the transformers patch,
# or eval behavior), then re-run the smoke tests in the README. When bumping
# LIBERO_PLUS_REF, also revisit _PLUS_ASSETS_REVISION in libero_eval/benchmarks.py.
OPENPI_REF=15a9616a00943ada6c20a0f158e3adb39df2ccac
LIBERO_PRO_REF=eafdb809426b13153aa1e4c42d6601844217dfec
LIBERO_PLUS_REF=4976dc30028e805ff8094b55501d532c48fec182
OPENVLA_OFT_REF=e4287e94541f459edc4feabc4e181f537cd569a8

WITH_CHECKPOINT=0
WITH_OPENVLA_OFT=0
for arg in "$@"; do
    case "$arg" in
        --with-checkpoint) WITH_CHECKPOINT=1 ;;
        --with-openvla-oft) WITH_OPENVLA_OFT=1 ;;
        *) echo "unknown setup option: $arg"; exit 2 ;;
    esac
done

command -v uv >/dev/null || { echo "uv not found — install from https://docs.astral.sh/uv/"; exit 1; }
mkdir -p "$THIRD_PARTY"

echo "== 1/5 third_party checkouts =="
if [ ! -e "$OPENPI_DIR" ]; then
    git clone https://github.com/Physical-Intelligence/openpi "$OPENPI_DIR"
    git -C "$OPENPI_DIR" checkout --quiet "$OPENPI_REF"
fi
git -C "$OPENPI_DIR" submodule update --init third_party/libero
if [ ! -e "$LIBERO_PRO_DIR" ]; then
    git clone https://github.com/Zxy-MLlab/LIBERO-PRO "$LIBERO_PRO_DIR"
    git -C "$LIBERO_PRO_DIR" checkout --quiet "$LIBERO_PRO_REF"
fi
if [ ! -e "$LIBERO_PLUS_DIR" ]; then
    git clone https://github.com/sylvestf/LIBERO-plus "$LIBERO_PLUS_DIR"
    git -C "$LIBERO_PLUS_DIR" checkout --quiet "$LIBERO_PLUS_REF"
fi
if [ "$WITH_OPENVLA_OFT" -eq 1 ] && [ ! -e "$OPENVLA_OFT_DIR" ]; then
    git clone https://github.com/moojink/openvla-oft "$OPENVLA_OFT_DIR"
    git -C "$OPENVLA_OFT_DIR" checkout --quiet "$OPENVLA_OFT_REF"
fi

echo "== 2/5 validator env (uv sync) =="
(cd "$VALIDATOR_ROOT" && uv sync)

echo "== 3/5 openpi server env (uv sync, Python >=3.11, JAX/torch CUDA) =="
(cd "$OPENPI_DIR" && GIT_LFS_SKIP_SMUDGE=1 uv sync)
# PyPI currently resolves openpi's torch==2.7.1 pin to a CUDA 12.6 wheel,
# which supports Ada (RTX 4090, sm_89) but not Blackwell (RTX 5090, sm_120).
# The official CUDA 12.8 build supports both architectures. Keep the upstream
# torch/torchvision versions so this only changes their bundled CUDA runtime.
VIRTUAL_ENV="$OPENPI_DIR/.venv" uv pip install \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --reinstall-package torch --reinstall-package torchvision
# openpi's PyTorch path needs its patched transformers modeling files; without
# them serving a model.safetensors checkpoint fails with a loud ValueError.
TRANSFORMERS_PKG="$("$OPENPI_DIR/.venv/bin/python" - <<'EOF'
import pathlib, transformers
print(pathlib.Path(transformers.__file__).parent)
EOF
)"
cp -r "$OPENPI_DIR/src/openpi/models_pytorch/transformers_replace/." "$TRANSFORMERS_PKG/"

echo "== 4/5 LIBERO client env (Python 3.8 + MuJoCo/robosuite) =="
cd "$OPENPI_DIR"
[ -d examples/libero/.venv ] || uv venv --python 3.8 examples/libero/.venv
VIRTUAL_ENV="$OPENPI_DIR/examples/libero/.venv" uv pip sync \
    examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
VIRTUAL_ENV="$OPENPI_DIR/examples/libero/.venv" uv pip install -e packages/openpi-client -e third_party/libero
# LIBERO-plus imports these in env_wrapper for every task (uv pip sync above
# would remove them, hence installed after it). wand needs ImageMagick's
# MagickWand system library at import time.
VIRTUAL_ENV="$OPENPI_DIR/examples/libero/.venv" uv pip install wand scikit-image
if ! "$OPENPI_DIR/examples/libero/.venv/bin/python" -c "import wand.api" 2>/dev/null; then
    echo "MagickWand system library missing (needed for --benchmark libero_plus); trying apt ..."
    sudo apt-get install -y libmagickwand-dev \
        || echo "WARNING: could not install libmagickwand-dev — libero/libero_pro still work; install it manually before running libero_plus"
fi

if [ "$WITH_OPENVLA_OFT" -eq 1 ]; then
    echo "== optional OpenVLA-OFT server env (Python 3.10, CUDA 12.8) =="
    [ -d "$OPENVLA_OFT_DIR/.venv" ] || uv venv --python 3.10 "$OPENVLA_OFT_DIR/.venv"
    VIRTUAL_ENV="$OPENVLA_OFT_DIR/.venv" uv pip install \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128
    VIRTUAL_ENV="$OPENVLA_OFT_DIR/.venv" uv pip install \
        accelerate draccus==0.8.0 einops huggingface_hub json-numpy jsonlines matplotlib \
        peft==0.11.1 protobuf rich sentencepiece==0.1.99 timm==0.9.10 tokenizers==0.19.1 \
        'transformers @ git+https://github.com/moojink/transformers-openvla-oft.git' \
        tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow-metadata==1.15.0 \
        tensorflow_graphics==2021.12.3 \
        'dlimp @ git+https://github.com/moojink/dlimp_openvla' \
        diffusers==0.30.3 imageio msgpack wandb==0.18.7 websockets
    VIRTUAL_ENV="$OPENVLA_OFT_DIR/.venv" uv pip install -e "$OPENVLA_OFT_DIR" --no-deps \
        -e "$OPENPI_DIR/packages/openpi-client"
fi

echo "== 5/5 demo checkpoint (pi05_libero, 12.4 GB) =="
if [ "$WITH_CHECKPOINT" -eq 1 ]; then
    CACHE_ROOT="$HOME/.cache/openpi/openpi-assets"
    BASE_URL="https://storage.googleapis.com/openpi-assets"
    curl -s "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_libero/&fields=items(name,size)" \
      | python3 -c "import json,sys; [print(i['name'], i['size']) for i in json.load(sys.stdin).get('items',[])]" > /tmp/ckpt_sizes.txt
    # resume-capable loop until every file matches its GCS size
    for round in 1 2 3 4 5 6 7 8 9 10; do
      pending=0
      while read -r name size; do
        out="$CACHE_ROOT/$name"
        cur=$(stat -c %s "$out" 2>/dev/null || echo 0)
        [ "$cur" -eq "$size" ] && continue
        pending=$((pending+1))
        mkdir -p "$(dirname "$out")"
        curl -s -f --retry 3 --retry-delay 2 -C - -o "$out" "$BASE_URL/$name" || true
      done < /tmp/ckpt_sizes.txt
      [ "$pending" -eq 0 ] && break
    done
    echo "checkpoint at $CACHE_ROOT/checkpoints/pi05_libero"
else
    echo "skipped (pass --with-checkpoint to download)"
fi

echo "== setup complete =="
