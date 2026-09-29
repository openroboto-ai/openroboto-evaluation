#!/bin/bash
# Install the isolated Python runtime used by --benchmark axis_v2.0.
set -euo pipefail

VALIDATOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$VALIDATOR_ROOT/third_party/openpi}"
AXIS_RUNTIME_DIR="${AXIS_RUNTIME_DIR:-$VALIDATOR_ROOT/third_party/axis-runtime}"
AXIS_VENV="$AXIS_RUNTIME_DIR/.venv"

command -v uv >/dev/null || { echo "uv not found"; exit 1; }
if [ ! -d "$OPENPI_DIR/packages/openpi-client" ]; then
    echo "openpi-client source not found at $OPENPI_DIR/packages/openpi-client; run bash setup.sh first"
    exit 1
fi
OPENPI_DIR="$(cd "$OPENPI_DIR" && pwd -P)"
case "$OPENPI_DIR" in
    */references/*)
        echo "Refusing to install from read-only references checkout: $OPENPI_DIR" >&2
        echo "Pass OPENPI_DIR=/path/to/a/clean OpenPI clone outside references." >&2
        exit 1
        ;;
esac

mkdir -p "$AXIS_RUNTIME_DIR"
[ -d "$AXIS_VENV" ] || uv venv --python 3.11 "$AXIS_VENV"
VIRTUAL_ENV="$AXIS_VENV" uv pip install \
    'mujoco==3.11.0' \
    'numpy==1.26.4' \
    'pillow==12.3.0' \
    'zarr>=2.16,<4' \
    'httpx[http2,socks]>=0.27' \
    'pyyaml>=6.0' \
    -e "$OPENPI_DIR/packages/openpi-client"

(cd "$VALIDATOR_ROOT" && uv run --locked python -c \
    'from libero_eval.axis_release import prepare_release; print(prepare_release())')

if ! MUJOCO_GL=osmesa "$AXIS_VENV/bin/python" -c 'import mujoco, numpy, httpx, openpi_client; model=mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>"); renderer=mujoco.Renderer(model,32,32); renderer.render(); renderer.close(); print("AXIS runtime ready: MuJoCo", mujoco.__version__, "OSMesa")'; then
    echo "AXIS OSMesa renderer unavailable. Install the libosmesa6 system package, then rerun setup_axis.sh." >&2
    exit 1
fi
