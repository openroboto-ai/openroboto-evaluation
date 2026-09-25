#!/usr/bin/env bash
# Install the optional, pinned RoboDojo + Isaac Sim runtime used by run_eval.py.
set -euo pipefail

VALIDATOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBODOJO_DIR="${ROBODOJO_DIR:-${VALIDATOR_ROOT}/third_party/RoboDojo}"
ROBODOJO_REF="9226f48ea694b3f53db12d4922e8b1199f8d0891"
ROBODOJO_DATA_REF="1a3c4c334aef294c31d7a0190d8d6dff68df78e0"
WITH_PI0=0
SKIP_ASSETS=0

for arg in "$@"; do
    case "$arg" in
        --with-pi0) WITH_PI0=1 ;;
        --skip-assets) SKIP_ASSETS=1 ;;
        *) echo "unknown RoboDojo setup option: $arg"; exit 2 ;;
    esac
done

command -v git >/dev/null || { echo "git not found"; exit 1; }
command -v git-lfs >/dev/null || { echo "git-lfs not found"; exit 1; }
command -v conda >/dev/null || { echo "conda not found"; exit 1; }
command -v uv >/dev/null || { echo "uv not found"; exit 1; }
for pkg in cmake build-essential ffmpeg; do
    dpkg -s "$pkg" >/dev/null 2>&1 || {
        echo "missing system package '$pkg'; install it before rerunning setup_robodojo.sh"
        exit 1
    }
done

mkdir -p "$(dirname "$ROBODOJO_DIR")"
if [ ! -e "$ROBODOJO_DIR" ]; then
    git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git "$ROBODOJO_DIR"
fi
git -C "$ROBODOJO_DIR" fetch origin "$ROBODOJO_REF"
git -C "$ROBODOJO_DIR" checkout --detach "$ROBODOJO_REF"
# Deliberately omit upstream install.sh's --remote: the parent commit's exact
# submodule SHAs are part of the benchmark definition.
git -C "$ROBODOJO_DIR" submodule sync --recursive
git -C "$ROBODOJO_DIR" submodule update --init --recursive --progress

eval "$(conda shell.bash hook)"
# XPolicyLab's policy launchers resolve deploy.yml with `<conda base>/bin/python`
# before activating the policy venv, so PyYAML must also exist in conda base.
CONDA_BASE_PYTHON="$(conda info --base)/bin/python"
uv pip install --python "$CONDA_BASE_PYTHON" 'pyyaml>=6'
if ! conda env list | grep -q '^RoboDojo '; then
    conda create -n RoboDojo python=3.11 -y
fi
conda activate RoboDojo
export PIP_USER=0
export PYTHONNOUSERSITE=1
python -m pip install -r "$ROBODOJO_DIR/scripts/requirements.txt"
python -m pip install opencv-python-headless==4.11.0.86 pillow matplotlib scipy==1.15.3 scikit-learn numpy==1.26.0

# Isaac Sim, IsaacLab, and cuRobo. Starting here avoids the upstream
# `submodules` step, which otherwise advances every pinned submodule to remote HEAD.
bash "$ROBODOJO_DIR/scripts/install.sh" --from isaacsim
PI05_ENV="$ROBODOJO_DIR/XPolicyLab/policy/Pi_05/openpi/.venv"
VIRTUAL_ENV="$PI05_ENV" bash "$ROBODOJO_DIR/XPolicyLab/policy/Pi_05/install.sh"
if [ "$WITH_PI0" -eq 1 ]; then
    PI0_ENV="$ROBODOJO_DIR/XPolicyLab/policy/Pi_0/openpi/.venv"
    VIRTUAL_ENV="$PI0_ENV" bash "$ROBODOJO_DIR/XPolicyLab/policy/Pi_0/install.sh"
fi

if [ "$SKIP_ASSETS" -eq 0 ]; then
    ASSET_CACHE="$ROBODOJO_DIR/.cache/robodojo_assets_repo"
    for attempt in 1 2 3 4 5; do
        if HF_REVISION="$ROBODOJO_DATA_REF" bash "$ROBODOJO_DIR/scripts/init_assets.sh"; then
            break
        fi
        if [ "$attempt" -eq 5 ]; then
            echo "RoboDojo asset download failed after $attempt attempts"
            exit 1
        fi
        # `git clone` can leave an empty/non-git target after a TLS failure;
        # preserve it for diagnosis so the next attempt has a clean path.
        if [ -d "$ASSET_CACHE" ] && [ ! -d "$ASSET_CACHE/.git" ]; then
            mv "$ASSET_CACHE" "${ASSET_CACHE}.partial.$(date +%Y%m%d_%H%M%S)"
        fi
        echo "asset download attempt $attempt failed; retrying in 5s"
        sleep 5
    done
else
    echo "RoboDojo assets skipped; install them before a real evaluation."
fi

if [ -d "$ROBODOJO_DIR/Assets/Robots" ]; then
    # CuRobo consumes generated configs with absolute mesh/URDF paths. The HF
    # asset bundle intentionally ships only *_tmp.yml templates.
    (cd "$ROBODOJO_DIR" && python utils/update_embodiment_config_path.py)
fi

OMNI_KIT_ACCEPT_EULA=YES bash "$ROBODOJO_DIR/scripts/robodojo.sh" doctor
echo "RoboDojo runtime ready at $ROBODOJO_DIR"
