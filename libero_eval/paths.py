"""Repository layout for the standalone validator.

Everything external lives under <validator>/third_party by default (created by
setup.sh — clones, or symlinks to existing checkouts). Both locations can be
overridden via environment variables so the pipeline never assumes a specific
parent workspace:

    OPENPI_DIR       openpi checkout (policy server code + both venvs inside)
    LIBERO_PRO_DIR   LIBERO-PRO checkout (benchmark fork + perturbation assets)
    LIBERO_PLUS_DIR  LIBERO-plus checkout (benchmark fork; bddl/init in-repo,
                     asset pack downloaded on first use)
    OPENVLA_OFT_DIR  OpenVLA-OFT checkout (optional policy server runtime)
"""

import os
import pathlib

VALIDATOR_ROOT = pathlib.Path(__file__).resolve().parent.parent
THIRD_PARTY = VALIDATOR_ROOT / "third_party"

OPENPI_DIR = pathlib.Path(os.environ.get("OPENPI_DIR", THIRD_PARTY / "openpi")).expanduser()
LIBERO_PRO_DIR = pathlib.Path(os.environ.get("LIBERO_PRO_DIR", THIRD_PARTY / "LIBERO-PRO")).expanduser()
LIBERO_PLUS_DIR = pathlib.Path(os.environ.get("LIBERO_PLUS_DIR", THIRD_PARTY / "LIBERO-plus")).expanduser()
OPENVLA_OFT_DIR = pathlib.Path(os.environ.get("OPENVLA_OFT_DIR", THIRD_PARTY / "openvla-oft")).expanduser()

# The two heavyweight Python environments live inside the openpi checkout
# (built by setup.sh): the server env via `uv sync`, the LIBERO client env as a
# Python 3.8 venv (LIBERO/robosuite do not run on modern Python).
SERVER_VENV_PY = OPENPI_DIR / ".venv" / "bin" / "python"
CLIENT_VENV_PY = OPENPI_DIR / "examples" / "libero" / ".venv" / "bin" / "python"
OPENVLA_OFT_VENV_PY = OPENVLA_OFT_DIR / ".venv" / "bin" / "python"
