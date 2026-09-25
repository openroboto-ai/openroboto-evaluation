"""Bind both OpenPI packages to the selected independent checkout."""

from __future__ import annotations

import pathlib
import sys


def bind_openpi_sources(root: pathlib.Path) -> pathlib.Path:
    root = root.expanduser().resolve()
    if "references" in root.parts:
        raise ValueError("AXIS must use an independent OpenPI checkout outside references")
    package_roots = {
        "openpi": root / "src",
        "openpi_client": root / "packages/openpi-client/src",
    }
    for package, parent in package_roots.items():
        package_dir = parent / package
        if not package_dir.is_dir():
            raise FileNotFoundError(f"{package} source not found at {package_dir}")
        loaded = sys.modules.get(package)
        loaded_file = getattr(loaded, "__file__", None)
        if loaded is not None and (
            loaded_file is None or package_dir not in pathlib.Path(loaded_file).resolve().parents
        ):
            raise RuntimeError(f"{package} was already imported from a different checkout; start a fresh process")
    # Shared dependency venvs can retain editable installs pointing at references.
    # Pin the client as well as the model package before any OpenPI import.
    sys.path[:0] = [str(parent) for parent in package_roots.values()] + [str(root)]
    return root
