"""Public metadata retains frozen identities without disclosing host paths."""
import hashlib
import json
import pathlib

import pytest

COMPONENT_ROOT = pathlib.Path(__file__).resolve().parents[1] / "libero_eval/axis_components"


def test_sanitized_profile_accepts_source_and_distribution_hashes():
    pytest.importorskip("mujoco")
    from libero_eval.axis_perturbations import load_visual_profile
    sources = json.loads((COMPONENT_ROOT / "SOURCES.json").read_bytes())
    filename = "profiles/franka_v6.json"
    current = sources["profiles"][filename]
    original = sources["profile_source_hashes"][filename]
    assert current != original
    assert load_visual_profile(current) == load_visual_profile(original)
    with pytest.raises(ValueError):
        load_visual_profile(None)
    with pytest.raises(ValueError):
        load_visual_profile("0" * 64)


def test_sanitized_vendor_hashes_and_paths():
    sources = json.loads((COMPONENT_ROOT / "SOURCES.json").read_bytes())
    for filename, digest in {**sources["files"], **sources["profiles"]}.items():
        assert hashlib.sha256((COMPONENT_ROOT / filename).read_bytes()).hexdigest() == digest
    for path in COMPONENT_ROOT.rglob("*.json"):
        raw = path.read_text()
        for prefix in ("/home/", "/Users/", "/data2/"):
            assert prefix not in raw, str(path.relative_to(COMPONENT_ROOT))


def test_redacted_scene_library_resolves_with_its_source_digest():
    pytest.importorskip("numpy")
    from libero_eval import axis_scene

    sources = json.loads((COMPONENT_ROOT / "SOURCES.json").read_bytes())
    camera = json.loads((COMPONENT_ROOT / "config/cameras/franka.json").read_bytes())
    library = camera["render_profile"]["scene"]
    assert sources["files"][library["library_file"]] != library["library_sha256"]
    visual = {
        "mode": "official_franka_components_v2",
        "camera_config_sha256": axis_scene.CAMERA_CONFIG_SHA256,
        "components": {
            "arena": True,
            "background": True,
            "front_camera": True,
            "wrist_camera": False,
            "surfaces": {"floor": [], "table": [], "wall": []},
        },
        "global_seed": 42,
        "attempt_id": 0,
        "variant_id": 3802,
        "replica_id": 0,
    }
    config, _ = axis_scene.resolve_profile(visual, 22, "Grab Can")
    assert config["render_profile"]["scene"]["library_sha256"] == library["library_sha256"]
