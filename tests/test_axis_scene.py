import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")
mujoco = pytest.importorskip("mujoco")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_scene import CAMERA_CONFIG_SHA256, resolve_profile
from axis_perturbations import RESET_REVISION, install_randomization_assets, sample_material
from axis_runtime import AxisEnvironment


def visual_config(variant=0):
    return {
        "mode": "official_franka_v6",
        "camera_config_sha256": CAMERA_CONFIG_SHA256,
        "global_seed": 42,
        "attempt_id": 2206084,
        "variant_id": variant,
        "replica_id": 0,
    }


def test_complete_scene_and_materials_match_released_attempt():
    record = json.loads((ROOT / "tests/fixtures/axis_randomization/task2286_scene.json").read_bytes())
    config, profile = resolve_profile(visual_config(), record["task_id"], record["task_name"])
    assert config["render_profile"]["scene"] == record["scene"]
    randomization = record["randomization_config"]
    for surface in ("table", "floor", "wall"):
        actual = sample_material(profile, surface, randomization[surface + "_material_seed"])
        expected = randomization["texture"]["surface_material_samples"][surface]
        assert actual["asset"]["sha256"] == expected["sha256"]
        np.testing.assert_array_equal(actual["material"], expected["material"])


def test_all_theme_assets_have_the_exact_upstream_bytes():
    root = ROOT / "libero_eval/axis_components"
    catalog = json.loads((root / "manifests/scene_materials.json").read_bytes())
    assert len(catalog) == 52
    for entry in catalog.values():
        assert (
            hashlib.sha256((root / "assets" / (entry["sha256"] + ".jpg")).read_bytes()).hexdigest() == entry["sha256"]
        )


def scene_fixture(tmp_path):
    xml = """<mujoco><compiler angle="radian"/><option timestep=".002"/>
    <worldbody><geom name="original_floor" type="plane" size="3 3 .01"/>
    <body name="franka/" pos="0 0 0"><geom type="sphere" size=".03" group="2"/>
      <body name="panda_link8" pos=".3 0 .3"><joint name="hinge" damping="1"/>
        <geom type="box" size=".03 .02 .01" group="2"/>
      </body>
    </body>
    <body name="box" pos=".4 0 .02"><freejoint name="box_joint"/>
      <geom name="box_visual" type="box" size=".02 .02 .02" group="1" rgba="1 0 0 1"/>
    </body></worldbody><actuator><position name="hinge" joint="hinge" kp="1"/></actuator></mujoco>"""
    config = {
        "schema_version": 1,
        "upstream_reset_revision": RESET_REVISION,
        "submit_nonce": "00123456",
        "domain_randomization": {},
        "object_order": [],
        "visual": visual_config(),
    }
    payload = {
        "id": 2286,
        "name": "Put the Camera on the Tray",
        "mjcf_xml": xml,
        "checker_config": {"type": "JointThresholdChecker", "jointName": "hinge", "threshold": 1, "comparison": "gt"},
        "initial_state": None,
        "official_randomization": config,
    }
    runtime = {
        "control_period_s": 0.02,
        "observation_joint_order": ["hinge"],
        "image_size": 256,
        "image_width": 640,
        "image_height": 360,
        "camera": "frontview",
        "settle_control_steps": 1,
    }
    assets = tmp_path / "assets"
    (assets / "scenes").mkdir(parents=True)
    scene = assets / "scenes/task.xml"
    scene.write_text(xml)
    install_randomization_assets(config, assets)
    return scene, payload, runtime


def test_full_native_room_repeats_and_never_changes_physics(tmp_path):
    scene, payload, runtime = scene_fixture(tmp_path)
    baseline = copy.deepcopy(payload)
    baseline["official_randomization"]["visual"] = None
    env = AxisEnvironment(scene, payload, runtime)
    physical = AxisEnvironment(scene, baseline, {**runtime, "camera": "unused"})
    try:
        env.reset()
        physical.reset()
        np.testing.assert_array_equal(env.data.qpos, physical.data.qpos)
        for field in ("body_mass", "geom_friction", "geom_contype", "geom_conaffinity", "jnt_type", "geom_size"):
            np.testing.assert_array_equal(getattr(env.model, field), getattr(physical.model, field))
        assert env.axis_scene.model.ngeom > env.model.ngeom + 30
        visual = env.reset_randomization["visual"]
        assert visual["scene"]["recipe_count"] == 8640
        assert visual["xml_metadata"]["collision_geoms_added"] == 0
        first = env.render()
        wrist = env.render(camera="wrist")
        assert first.shape == wrist.shape == (360, 640, 3)
        assert not np.array_equal(first, wrist)
        env.reset()
        np.testing.assert_array_equal(first, env.render())
        for _ in range(3):
            env.step([0.2])
            physical.step([0.2])
            env.render()
        np.testing.assert_array_equal(env.data.qpos, physical.data.qpos)
        np.testing.assert_array_equal(env.data.qvel, physical.data.qvel)
        np.testing.assert_array_equal(env.axis_scene.data.qpos, env.data.qpos + env.axis_scene.qpos_offset)
    finally:
        env.close()
        physical.close()
    variant = copy.deepcopy(payload)
    variant["official_randomization"]["visual"]["variant_id"] = 1
    other = AxisEnvironment(scene, variant, runtime)
    try:
        other.reset()
        assert not np.array_equal(first, other.render())
        assert other.reset_randomization["visual"]["scene"]["recipe_sha256"] != visual["scene"]["recipe_sha256"]
    finally:
        other.close()


def test_full_profile_rejects_unpublished_camera_replica_and_bad_hash():
    for updates, match in [({"replica_id": 4}, "four randomized"), ({"camera_config_sha256": "0" * 64}, "unpinned")]:
        with pytest.raises(ValueError, match=match):
            resolve_profile({**visual_config(), **updates}, 2286, "Put the Camera on the Tray")


def test_legacy_wrist_frame_is_rejected_without_adjusting_the_robot(tmp_path):
    scene, payload, runtime = scene_fixture(tmp_path)
    root = ET.fromstring(scene.read_text())
    base = root.find(".//body[@name='franka/']")
    hand = base.find("body")
    base.remove(hand)
    link7 = ET.SubElement(base, "body", name="panda_link7")
    link7.append(hand)
    hand.set("name", "panda_hand")
    hand.set("pos", "0 0 .107")
    hand.set("quat", "1 0 0 0")
    scene.write_text(ET.tostring(root, encoding="unicode"))
    original = scene.read_bytes()
    with pytest.raises(ValueError, match="source quaternion mismatch"):
        AxisEnvironment(scene, payload, runtime)
    assert scene.read_bytes() == original


def test_camera_injection_leaves_included_files_untouched(tmp_path):
    scene, payload, runtime = scene_fixture(tmp_path)
    include = scene.with_name("original.xml")
    include.write_bytes(scene.read_bytes())
    scene.write_text('<mujoco><include file="original.xml"/></mujoco>')
    before = {p: p.read_bytes() for p in (scene, include)}
    env = AxisEnvironment(scene, payload, runtime)
    try:
        env.reset()
        assert env.render().shape == (360, 640, 3)
    finally:
        env.close()
    assert {p: p.read_bytes() for p in before} == before
