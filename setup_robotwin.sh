#!/bin/bash
# Build the two isolated runtimes used by LingBot-VLA 2.0 + RoboTwin 2.0.
set -euo pipefail

VALIDATOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="$VALIDATOR_ROOT/third_party"
LINGBOT_DIR="${LINGBOT_VLA_V2_DIR:-$THIRD_PARTY/lingbot-vla-v2}"
ROBOTWIN_DIR="${ROBOTWIN_DIR:-$THIRD_PARTY/RoboTwin}"
QWEN3_DIR="${QWEN3_VL_DIR:-$THIRD_PARTY/Qwen3-VL-4B-Instruct}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-3.10}"
ROBOTWIN_LD_LIBRARY_PATH="${ROBOTWIN_LD_LIBRARY_PATH:-}"

# These revisions form one tested protocol unit.  RoboTwin's newer XPolicyLab
# layout is intentionally not mixed with LingBot's legacy official client.
LINGBOT_REF=951475ae1b1d87553e7dc47c97b53a3d695c0d13
ROBOTWIN_REF=13c3c47ff4312dd62484bcd51be034af55c062d1
# Latest RoboTwin2.0 dataset revision before the LingBot checkpoint/results
# release. The installed archives are byte-identical, but pinning prevents a
# future dataset-main update from silently changing the benchmark.
ROBOTWIN_ASSETS_REF=9dc9299c163db059931898a9f0852098a61155a1
QWEN3_REF=ebb281ec70b05090aa6165b016eac8ec08e71b17
CHECKPOINT_REF=0451855729ec904f970600e0aec8b84661423afe

WITH_ASSETS=0
WITH_CHECKPOINT=0
for arg in "$@"; do
    case "$arg" in
        --with-assets) WITH_ASSETS=1 ;;
        --with-checkpoint) WITH_CHECKPOINT=1 ;;
        *) echo "unknown setup option: $arg"; exit 2 ;;
    esac
done

command -v git >/dev/null || { echo "git not found"; exit 1; }
command -v uv >/dev/null || { echo "uv not found — install from https://docs.astral.sh/uv/"; exit 1; }
mkdir -p "$THIRD_PARTY"

checkout_pinned() {
    local url="$1"
    local path="$2"
    local revision="$3"
    if [ ! -e "$path" ]; then
        GIT_LFS_SKIP_SMUDGE=1 git clone "$url" "$path"
    fi
    if [ ! -d "$path/.git" ]; then
        echo "$path exists but is not a git checkout"
        exit 1
    fi
    git -C "$path" fetch origin "$revision"
    git -C "$path" checkout --quiet "$revision"
}

echo "== 1/5 pinned upstream checkouts =="
checkout_pinned https://github.com/Robbyant/lingbot-vla-v2 "$LINGBOT_DIR" "$LINGBOT_REF"
checkout_pinned https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_DIR" "$ROBOTWIN_REF"

echo "== 2/5 LingBot-VLA 2.0 inference env =="
[ -x "$LINGBOT_DIR/.venv/bin/python" ] || uv venv --python 3.12 "$LINGBOT_DIR/.venv"
VIRTUAL_ENV="$LINGBOT_DIR/.venv" uv pip install -r "$LINGBOT_DIR/requirements.txt"
VIRTUAL_ENV="$LINGBOT_DIR/.venv" uv pip install setuptools wheel
VIRTUAL_ENV="$LINGBOT_DIR/.venv" uv pip install numpydantic==1.9.0 --no-deps
VIRTUAL_ENV="$LINGBOT_DIR/.venv" MAX_JOBS="${MAX_JOBS:-8}" \
    uv pip install flash-attn==2.8.3 --no-build-isolation
VIRTUAL_ENV="$LINGBOT_DIR/.venv" uv pip install --no-deps \
    "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
VIRTUAL_ENV="$LINGBOT_DIR/.venv" uv pip install -e "$LINGBOT_DIR" --no-deps

echo "== 3/5 pinned Qwen3-VL processor files =="
"$LINGBOT_DIR/.venv/bin/python" - "$QWEN3_DIR" "$QWEN3_REF" <<'PY'
import pathlib
import sys
from huggingface_hub import snapshot_download

destination = pathlib.Path(sys.argv[1])
snapshot_download(
    repo_id="Qwen/Qwen3-VL-4B-Instruct",
    revision=sys.argv[2],
    local_dir=destination,
    ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.onnx"],
)
PY

echo "== 4/5 RoboTwin simulator env =="
[ -x "$ROBOTWIN_DIR/.venv/bin/python" ] || uv venv --python "$ROBOTWIN_PYTHON" "$ROBOTWIN_DIR/.venv"

# A Python executable taken directly from a conda package cache may depend on
# the conda installation's libexpat while the host exports /lib first.  Infer
# that library directory for this uncommon but useful offline-Python setup.
ROBOTWIN_BASE_PYTHON="$(readlink -f "$ROBOTWIN_DIR/.venv/bin/python")"
if [ -z "$ROBOTWIN_LD_LIBRARY_PATH" ] && [[ "$ROBOTWIN_BASE_PYTHON" == */pkgs/*/bin/python* ]]; then
    ROBOTWIN_CONDA_ROOT="${ROBOTWIN_BASE_PYTHON%%/pkgs/*}"
    if [ -e "$ROBOTWIN_CONDA_ROOT/lib/libexpat.so.1" ]; then
        ROBOTWIN_LD_LIBRARY_PATH="$ROBOTWIN_CONDA_ROOT/lib"
    fi
fi
run_robotwin_python() {
    if [ -n "$ROBOTWIN_LD_LIBRARY_PATH" ]; then
        LD_LIBRARY_PATH="$ROBOTWIN_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$ROBOTWIN_DIR/.venv/bin/python" "$@"
    else
        "$ROBOTWIN_DIR/.venv/bin/python" "$@"
    fi
}
printf '%s\n' "$ROBOTWIN_LD_LIBRARY_PATH" > "$ROBOTWIN_DIR/.runtime_library_path"

ROBOTWIN_REQUIREMENTS="$(mktemp /tmp/robotwin-eval-requirements.XXXXXX)"
# azure==4.0.0 is a retired meta-package that intentionally aborts install;
# the simulator/evaluation path does not import it.
sed '/^azure==/d' "$ROBOTWIN_DIR/script/requirements.txt" > "$ROBOTWIN_REQUIREMENTS"
VIRTUAL_ENV="$ROBOTWIN_DIR/.venv" uv pip install -r "$ROBOTWIN_REQUIREMENTS"
rm -f "$ROBOTWIN_REQUIREMENTS"
VIRTUAL_ENV="$ROBOTWIN_DIR/.venv" uv pip install \
    setuptools wheel ninja websockets==15.0.1 msgpack==1.1.1
VIRTUAL_ENV="$ROBOTWIN_DIR/.venv" uv pip install \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

CUROBO_DIR="$ROBOTWIN_DIR/envs/curobo"
if [ ! -e "$CUROBO_DIR" ]; then
    git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
fi
if ! run_robotwin_python -c 'import curobo' >/dev/null 2>&1; then
    VIRTUAL_ENV="$ROBOTWIN_DIR/.venv" uv pip install -e "$CUROBO_DIR" --no-build-isolation
fi
VIRTUAL_ENV="$ROBOTWIN_DIR/.venv" uv pip install warp-lang==1.12.0 setuptools==69.5.1

# Apply the two fixes from RoboTwin's official installer inside this isolated
# venv only.  They are idempotent substitutions at pinned upstream lines.
SAPIEN_PACKAGE="$(run_robotwin_python -c 'import pathlib,sapien; print(pathlib.Path(sapien.__file__).parent)')"
MPLIB_PACKAGE="$(run_robotwin_python -c 'import pathlib,mplib; print(pathlib.Path(mplib.__file__).parent)')"
sed -i -E 's/with open\((urdf_file|srdf_file), "r"\) as f:/with open(\1, "r", encoding="utf-8") as f:/' \
    "$SAPIEN_PACKAGE/wrapper/urdf_loader.py"
sed -i -E 's/srdf_file = urdf_file\[:-4\] \+ "srdf"/srdf_file = urdf_file[:-4] + ".srdf"/' \
    "$SAPIEN_PACKAGE/wrapper/urdf_loader.py"
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )or collide (or not within_joint_limit:)/\1\2/' \
    "$MPLIB_PACKAGE/planner.py"

if [ "$WITH_ASSETS" -eq 1 ]; then
    echo "== downloading RoboTwin assets =="
    if [ ! -d "$ROBOTWIN_DIR/assets/background_texture" ] || \
       [ ! -d "$ROBOTWIN_DIR/assets/embodiments" ] || \
       [ ! -d "$ROBOTWIN_DIR/assets/objects" ]; then
        # Keep the upstream archive selection, but freeze the otherwise
        # mutable Hugging Face dataset revision.
        run_robotwin_python - "$ROBOTWIN_DIR" "$ROBOTWIN_ASSETS_REF" <<'PY'
import pathlib
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    revision=sys.argv[2],
    allow_patterns=["background_texture.zip", "embodiments.zip", "objects.zip"],
    local_dir=pathlib.Path(sys.argv[1]),
    repo_type="dataset",
)
PY
        for archive in background_texture embodiments objects; do
            unzip -q -o "$ROBOTWIN_DIR/${archive}.zip" -d "$ROBOTWIN_DIR/assets"
            rm -f "$ROBOTWIN_DIR/${archive}.zip"
        done
    fi
    (cd "$ROBOTWIN_DIR" && run_robotwin_python script/update_embodiment_config_path.py)
else
    echo "RoboTwin assets skipped (pass --with-assets before evaluation)"
fi

echo "== 5/5 official LingBot checkpoint =="
if [ "$WITH_CHECKPOINT" -eq 1 ]; then
    CHECKPOINT_ROOT="$VALIDATOR_ROOT/hf_models/robbyant__lingbot-vla-v2-6b-robotwin@${CHECKPOINT_REF:0:12}"
    HFD_BIN="${HFD_BIN:-}"
    if [ -z "$HFD_BIN" ] && [ -f "$VALIDATOR_ROOT/hfd.sh" ]; then
        HFD_BIN="$VALIDATOR_ROOT/hfd.sh"
    elif [ -z "$HFD_BIN" ] && command -v hfd.sh >/dev/null; then
        HFD_BIN="$(command -v hfd.sh)"
    fi
    if [ -n "$HFD_BIN" ]; then
        bash "$HFD_BIN" robbyant/lingbot-vla-v2-6b-robotwin \
            --local-dir "$CHECKPOINT_ROOT" --revision "$CHECKPOINT_REF" -x 10 -j 6
    else
        "$LINGBOT_DIR/.venv/bin/python" - "$CHECKPOINT_ROOT" "$CHECKPOINT_REF" <<'PY'
import pathlib
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="robbyant/lingbot-vla-v2-6b-robotwin",
    revision=sys.argv[2],
    local_dir=pathlib.Path(sys.argv[1]),
)
PY
    fi
    echo "checkpoint: $CHECKPOINT_ROOT/checkpoints/global_step_50000/hf_ckpt"
else
    echo "checkpoint skipped (pass --with-checkpoint to download ~25.5 GB)"
fi

echo "== RoboTwin setup complete =="
