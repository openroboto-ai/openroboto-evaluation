import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))
from axis_openpi_sources import bind_openpi_sources  # noqa: E402


class TestAxisOpenpiSources(unittest.TestCase):
    def fixture(self, root):
        for parent, package in ((root / "src", "openpi"), (root / "packages/openpi-client/src", "openpi_client")):
            folder = parent / package
            folder.mkdir(parents=True)
            (folder / "__init__.py").write_text("origin = 'pinned'\n")

    def test_shared_environment_client_cannot_shadow_the_selected_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary) / "pinned"
            self.fixture(root)
            shadow = pathlib.Path(temporary) / "shadow"
            for package in ("openpi", "openpi_client"):
                (shadow / package).mkdir(parents=True)
                (shadow / package / "__init__.py").write_text("origin = 'shadow'\n")
            source = (
                "import pathlib,sys; "
                "sys.path[:0]=sys.argv[1:3]; "
                "from axis_openpi_sources import bind_openpi_sources; "
                "bind_openpi_sources(pathlib.Path(sys.argv[3])); "
                "import openpi,openpi_client; "
                "assert openpi.origin == openpi_client.origin == 'pinned'"
            )
            subprocess.run(
                [sys.executable, "-c", source, str(ROOT / "libero_eval"), str(shadow), str(root)], check=True
            )

    def test_references_and_preloaded_foreign_clients_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary) / "pinned"
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, "outside references"):
                bind_openpi_sources(root / "references/openpi")
            with mock.patch.dict(sys.modules, {"openpi_client": types.SimpleNamespace(__file__="/foreign/client.py")}):
                with self.assertRaisesRegex(RuntimeError, "fresh process"):
                    bind_openpi_sources(root)
