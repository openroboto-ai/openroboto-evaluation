import copy
import hashlib
import json
import os
import pathlib
import sys
import types

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")
mujoco = pytest.importorskip("mujoco")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))
sys.path.insert(0, str(ROOT / "tools"))

from axis_perturbations import (
    AxisRandomizer,
    PROFILE_PATH,
    RESET_REVISION,
    install_randomization_assets,
    sample_cameras,
    sample_material,
)
from axis_runtime import AxisEnvironment, canonical_json_sha256, verify_task_payload
from axis_randomization import load_randomization_plan, build_trial_plan
from build_axis_randomized_release import build
from axis_task import run as run_axis_task

FIXTURES = ROOT / "tests/fixtures/axis_randomization"


def reset_config(cfg=None, nonce="00123456"):
    return {
        "schema_version": 1,
        "upstream_reset_revision": RESET_REVISION,
        "submit_nonce": nonce,
        "domain_randomization": cfg,
        "object_order": list((cfg or {}).get("objects", {})),
        "visual": None,
    }


def tiny_model():
    return mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="box" pos="0 0 1"><freejoint/><geom type="box" size=".02 .02 .02"/></body>
      <body name="ball" pos="1 0 1"><freejoint/><geom type="sphere" size=".02"/></body>
      <body name="franka/base" pos="2 0 1"><freejoint/><geom type="sphere" size=".02"/></body>
    </worldbody></mujoco>""")


def randomizer(model, cfg):
    return AxisRandomizer(
        model, cfg, asset_root=ROOT / ".cache/axis/assets", task_id=1952, task_name="fixture", width=640, height=360
    )


def test_backend_default_is_reproducible_and_empty_disables_it():
    model = tiny_model()
    data = mujoco.MjData(model)
    adapter = randomizer(model, reset_config())
    adapter.apply(data, None)
    np.testing.assert_array_equal(data.qpos[:2], [-0.011766695650294423, 0.02972629074938596])
    np.testing.assert_array_equal(data.qpos[14:17], [2, 0, 1])
    first = data.qpos.copy()
    mujoco.mj_resetData(model, data)
    adapter.apply(data, None)
    np.testing.assert_array_equal(data.qpos, first)
    mujoco.mj_resetData(model, data)
    randomizer(model, reset_config({})).apply(data, None)
    np.testing.assert_array_equal(data.qpos, model.qpos0)
    randomizer(model, reset_config(nonce="00123457")).apply(data, None)
    assert not np.array_equal(data.qpos, first)


def test_explicit_object_order_survives_canonical_json_and_swap_rotates_full_pose():
    model = tiny_model()
    cfg = {
        "objects": {
            "box": {"pos_delta": [[-0.01, 0.01], [0, 0], [0, 0]], "rot_delta": [[0, 0], [0, 0], [0.5, 0.5]]},
            "ball": {"pos_delta": [[-0.01, 0.01], [0, 0], [0, 0]]},
        },
        "swap_positions": {"pairs": [["box", "ball"]], "probability": 1},
    }
    config = reset_config(cfg)
    data1, data2 = mujoco.MjData(model), mujoco.MjData(model)
    randomizer(model, config).apply(data1, None)
    randomizer(model, json.loads(json.dumps(config, sort_keys=True))).apply(data2, None)
    np.testing.assert_array_equal(data1.qpos, data2.qpos)
    assert 0.99 < data1.qpos[0] < 1.01
    np.testing.assert_allclose(data1.qpos[10:14], [np.cos(0.25), 0, 0, np.sin(0.25)], atol=1e-15)


def test_explicit_mocap_translation_and_rotation():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="marker" mocap="true" pos="1 2 3"><geom type="sphere" size=".01"/></body>
    </worldbody></mujoco>""")
    config = reset_config({
        "objects": {"marker": {"pos_delta": [[0.1, 0.1], [0, 0], [0, 0]], "rot_delta": [[0, 0], [0, 0], [0.5, 0.5]]}}
    })
    data = mujoco.MjData(model)
    randomizer(model, config).apply(data, None)
    np.testing.assert_allclose(data.mocap_pos[0], [1.1, 2, 3], atol=1e-15)
    np.testing.assert_allclose(data.mocap_quat[0], [np.cos(0.25), 0, 0, np.sin(0.25)], atol=1e-15)


@pytest.mark.parametrize(
    "cfg, match",
    [
        ({"robots": {}}, "supports null"),
        ({"objects": {"absent": {}}}, "missing from"),
        ({"objects": {"box": {"pos_delta": [[2, 1], [0, 0], [0, 0]]}}}, "ordered ranges"),
        ({"objects": {"box": {"dof_pos": {}}}}, "unsupported"),
    ],
)
def test_invalid_or_unsupported_domain_config_fails(cfg, match):
    with pytest.raises(ValueError, match=match):
        randomizer(tiny_model(), reset_config(cfg))


@pytest.mark.parametrize("task_id", [1952, 757])
def test_camera_samples_equal_recorded_release_values(task_id):
    record = json.loads((FIXTURES / f"task{task_id}.json").read_text())
    profile = json.loads(PROFILE_PATH.read_text())
    for replica in range(4):
        front = record["camera_samples"][f"frontview_random_{replica:02}"]
        wrist = record["camera_samples"][f"wrist_random_{replica:02}"]
        g = front["geometry"]
        corners = np.asarray(g["table_world_corners"])
        geometry = {
            "base_position": g["base_world_position"],
            "base_quat_wxyz": g["base_world_quat_wxyz"],
            "table_center_xy": np.mean(corners[:, :2], axis=0),
            "table_top_z": corners[0, 2],
            "table_full_size": [np.ptp(corners[:, 0]), np.ptp(corners[:, 1]), 0.05],
            "wrist_base": wrist["base"],
        }
        samples = sample_cameras(profile, geometry, camera_seed=record["camera_seed"], replica_id=replica)
        for logical, expected, fields in [
            ("frontview", front, ["world_position", "world_quat_wxyz", "fovy_deg"]),
            ("wrist", wrist, ["position", "quat_wxyz", "fovy_deg"]),
        ]:
            for field in fields:
                np.testing.assert_array_equal(samples[logical]["resolved"][field], expected["resolved"][field])


def test_material_sampling_matches_upstream_record_and_assets_are_pinned():
    record = json.loads((FIXTURES / "task1952.json").read_text())
    profile = json.loads(PROFILE_PATH.read_text())
    for surface in ("table", "floor", "wall"):
        actual = sample_material(profile, surface, record["material_seeds"][surface])
        expected = record["surface_samples"][surface]
        assert actual["asset"]["sha256"] == expected["sha256"]
        np.testing.assert_array_equal(actual["material"], expected["material"])
    for asset in profile["materials"].values():
        path = PROFILE_PATH.parent.parent / "assets" / (asset["sha256"] + ".jpg")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]


def fixture_environment(tmp_path, *, visual=True, nonce="00123456"):
    record = json.loads((FIXTURES / "task1952.json").read_text())
    base = record["camera_samples"]["wrist_random_00"]["base"]
    xml = f'''<mujoco><option timestep=".002"/><visual><global offwidth="640" offheight="360"/></visual>
      <asset>
        <texture name="table_t" type="2d" builtin="flat" width="1181" height="1181"/>
        <texture name="floor_t" type="2d" builtin="flat" width="255" height="255"/>
        <texture name="wall_t" type="2d" builtin="flat" width="880" height="880"/>
        <material name="table_m" texture="table_t"/><material name="floor_m" texture="floor_t"/>
        <material name="wall_m" texture="wall_t"/>
      </asset><worldbody>
        <light pos="0 0 3"/>
        <geom name="table" type="box" pos=".35 0 .455" size=".7 .6 .025" material="table_m"/>
        <geom name="floor" type="plane" size="3 3 .01" material="floor_m"/>
        <geom name="wall" type="box" pos="-.9 0 1" size=".05 3 1" material="wall_m"/>
        <body name="franka/" pos="0 0 .479999993">
          <geom type="sphere" size=".03"/>
          <body name="hand" pos=".3 0 .3"><joint name="hinge" damping="1"/>
            <geom type="box" size=".03 .02 .01"/>
            <camera name="wrist" pos="{" ".join(map(str, base["position"]))}"
              quat="{" ".join(map(str, base["quat_wxyz"]))}" fovy="52"/>
          </body>
        </body>
        <body name="box" pos=".4 0 .51"><freejoint/><geom type="box" size=".02 .02 .02" rgba="1 0 0 1"/></body>
        <camera name="frontview" pos="1.2 0 1.5" xyaxes="0 1 0 -.8 0 .6" fovy="58"/>
      </worldbody><actuator><position name="hinge" joint="hinge" kp="1"/></actuator></mujoco>'''
    config = reset_config({}, nonce=nonce)
    if visual:
        profile = json.loads(PROFILE_PATH.read_text())
        config["visual"] = {
            "profile_sha256": hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
            "global_seed": 42,
            "attempt_id": 1925874,
            "variant_id": 0,
            "replica_id": 0,
            "front_camera": "frontview",
            "wrist_camera": "wrist",
            "reference_body": "franka/",
            "table_center_xy": [0.35, 0],
            "table_top_z": 0.48,
            "table_full_size": [1.4, 1.2, 0.05],
            "surfaces": {s: [s] for s in ["table", "floor", "wall"]},
            "assets": {key: f"official_textures/{asset['sha256']}.jpg" for key, asset in profile["materials"].items()},
        }
    asset_root = tmp_path / "assets"
    if visual == "full":
        from axis_scene import CAMERA_CONFIG_SHA256

        config["visual"] = {
            "mode": "official_franka_v6",
            "camera_config_sha256": CAMERA_CONFIG_SHA256,
            "global_seed": 42,
            "attempt_id": 1925874,
            "variant_id": 0,
            "replica_id": 0,
        }
        xml = (
            xml
            .replace('name="hand"', 'name="panda_link8"')
            .replace('pos="0 0 .479999993"', 'pos="0 0 0"')
            .replace('pos=".4 0 .51"', 'pos=".4 0 .03"')
            .replace('pos=".35 0 .455"', 'pos=".35 0 -.025"')
        )
    (asset_root / "scenes").mkdir(parents=True, exist_ok=True)
    install_randomization_assets(config, asset_root)
    scene = asset_root / "scenes/fixture.xml"
    scene.write_text(xml)
    payload = {
        "id": 1952,
        "name": record["task_name"],
        "mjcf_xml": xml,
        "checker_config": {"type": "JointThresholdChecker", "jointName": "hinge", "threshold": 1, "comparison": "gt"},
        "initial_state": None,
        "status": "fixture",
        "embodiment": "franka",
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
        "asset_base_url": "https://example.invalid/",
        "renderer_backend": "osmesa",
    }
    return scene, payload, runtime


def test_native_render_changes_pixels_is_repeatable_and_preserves_physics(tmp_path):
    scene, payload, runtime = fixture_environment(tmp_path)
    env = AxisEnvironment(scene, payload, runtime)
    try:
        mass, friction, light = (
            env.model.body_mass.copy(),
            env.model.geom_friction.copy(),
            env.model.light_diffuse.copy(),
        )
        env.reset()
        first, state = env.render(), env.data.qpos.copy()
        assert first.shape == (360, 640, 3) and first.std() > 10
        wrist_image = env.render(camera="wrist")
        assert wrist_image.shape == first.shape and not np.array_equal(wrist_image, first)
        env.reset()
        np.testing.assert_array_equal(env.render(), first)
        np.testing.assert_array_equal(env.data.qpos, state)
        np.testing.assert_array_equal(env.model.body_mass, mass)
        np.testing.assert_array_equal(env.model.geom_friction, friction)
        np.testing.assert_array_equal(env.model.light_diffuse, light)
        expected = json.loads((FIXTURES / "task1952.json").read_text())
        sample = env.reset_randomization["visual"]["camera_samples"]["frontview"]["resolved"]
        np.testing.assert_allclose(
            sample["world_position"],
            expected["camera_samples"]["frontview_random_00"]["resolved"]["world_position"],
            atol=1e-15,
        )
    finally:
        env.close()
    changed = copy.deepcopy(payload)
    changed["official_randomization"]["visual"]["replica_id"] = 1
    other = AxisEnvironment(scene, changed, runtime)
    try:
        other.reset()
        assert not np.array_equal(first, other.render())
        np.testing.assert_array_equal(state, other.data.qpos)
    finally:
        other.close()


def test_missing_surfaces_square_image_and_bad_hash_fail(tmp_path):
    scene, payload, runtime = fixture_environment(tmp_path)
    with pytest.raises(ValueError, match="16:9"):
        AxisEnvironment(scene, payload, {**runtime, "image_width": 256, "image_height": 256})
    payload["official_randomization"]["visual"]["surfaces"]["wall"] = []
    with pytest.raises(ValueError, match="missing surfaces"):
        AxisEnvironment(scene, payload, runtime)
    payload["official_randomization"]["visual"]["profile_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="profile hash mismatch"):
        AxisEnvironment(scene, payload, runtime)


def test_initial_success_is_an_error_not_a_resampled_seed(tmp_path):
    scene, payload, runtime = fixture_environment(tmp_path, visual=False)
    payload["checker_config"]["threshold"] = -1
    env = AxisEnvironment(scene, payload, runtime)
    try:
        with pytest.raises(ValueError, match="already satisfies"):
            env.reset()
    finally:
        env.close()


def test_frozen_hash_covers_official_config_and_variants(tmp_path):
    scene, payload, runtime = fixture_environment(tmp_path, visual=False)
    spec = {
        "task_id": payload["id"],
        "instruction": payload["name"],
        "mjcf_sha256": hashlib.sha256(payload["mjcf_xml"].encode()).hexdigest(),
        "checker_sha256": canonical_json_sha256(payload["checker_config"]),
        "initial_state_sha256": canonical_json_sha256(None),
    }
    with pytest.raises(ValueError, match="unpinned"):
        verify_task_payload(payload, spec)
    spec["official_randomization_sha256"] = canonical_json_sha256(payload["official_randomization"])
    assert verify_task_payload(payload, spec) == payload
    payload["official_randomization"]["submit_nonce"] = "00123457"
    with pytest.raises(ValueError, match="hash-mismatched"):
        verify_task_payload(payload, spec)


@pytest.mark.parametrize("visual", [False, True, "full"])
def test_builder_freezes_runnable_independent_release(tmp_path, visual, monkeypatch):
    scene, payload, runtime = fixture_environment(tmp_path, visual=visual)
    config = payload.pop("official_randomization")
    source = tmp_path / "source.json"
    specs = {
        "task_id": 1952,
        "instruction": payload["name"],
        "mjcf_sha256": hashlib.sha256(payload["mjcf_xml"].encode()).hexdigest(),
        "checker_sha256": canonical_json_sha256(payload["checker_config"]),
        "initial_state_sha256": canonical_json_sha256(None),
    }
    manifest = {
        "name": "axis_v90.0",
        "status": "runtime-ready",
        "protocol_revision": "source",
        "protocol": {"randomization": False},
        "runtime": runtime,
        "tasks": [specs],
    }
    source.write_text(json.dumps(manifest))
    (tmp_path / "source-tasks").mkdir()
    (tmp_path / "source-tasks/1952.json").write_text(json.dumps(payload))
    binding = {
        "schema_version": 1,
        "name": "axis_v90.1",
        "protocol_revision": "official-fixture",
        "namespace": "official-fixture",
        "tasks": [
            {
                "task_id": 1952,
                "source_payload_sha256": canonical_json_sha256(payload),
                "upstream_provenance": {"purpose": "test fixture"},
                "domain_randomization": None,
                "object_order": [],
                "visual": config["visual"],
            }
        ],
        "instances": [
            {
                "variant_id": f"nonce-{nonce}",
                "submit_nonce": nonce,
                "global_seed": 42,
                "attempt_id": 1925874,
                "render_variant_id": 0,
                "replica_id": index,
            }
            for index, nonce in enumerate(["00123456", "00123457"])
        ],
    }
    bindings = tmp_path / "bindings.json"
    bindings.write_text(json.dumps(binding))
    args = types.SimpleNamespace(
        source_manifest=str(source),
        bindings=str(bindings),
        output=str(tmp_path / "release"),
        cache_root=str(tmp_path / "cache"),
    )
    result = build(args)
    assert result["instances"] == 2
    plan = load_randomization_plan(
        tmp_path / "release/randomization.json",
        expected_benchmark="axis_v90.1",
        expected_protocol_revision="official-fixture",
        benchmark_task_specs={1952: specs},
    )
    trials = build_trial_plan(plan, task_id=1952, num_trials=20, seed=42)
    assert {t.payload["official_randomization"]["submit_nonce"] for t in trials} == {"00123456", "00123457"}
    assert source.read_text() == json.dumps(manifest)
    smoke = run_axis_task(
        types.SimpleNamespace(
            manifest=str(tmp_path / "release/benchmark.json"),
            task_id=1952,
            num_trials=4,
            randomization_manifest=str(tmp_path / "release/randomization.json"),
            randomization_seed=42,
            cache_root=str(tmp_path / "task-cache"),
            record_trials=0,
            asset_base_url=None,
            task_api_base_url=None,
            asset_fetch_workers=1,
            refresh_task=False,
            prepare_only=False,
            dry_run=True,
            smoke_control_steps=1,
            smoke_render_frames=1,
        )
    )
    assert smoke["status"] == "ok"
    details = smoke["variant_smoke"] or {"single": smoke}
    for value in details.values():
        assert value["smoke"]["official_randomization"]["submit_nonce"] in {"00123456", "00123457"}
        assert value["smoke"]["image_shape"] == [360, 640, 3]
    if visual:
        import axis_task

        requests = []

        class Policy:
            def infer(self, request):
                requests.append(request)
                assert request["observation/wrist_image"].shape == (360, 640, 3)
                assert not np.array_equal(request["observation/image"], request["observation/wrist_image"])
                return {"actions": request["observation/state"][None, :]}

        monkeypatch.setattr(axis_task, "_policy_client", lambda host, port: Policy())
        monkeypatch.setattr(axis_task, "_resize_image", lambda image, size: image)
        rollout = run_axis_task(
            types.SimpleNamespace(
                manifest=str(tmp_path / "release/benchmark.json"),
                task_id=1952,
                num_trials=2,
                randomization_manifest=str(tmp_path / "release/randomization.json"),
                randomization_seed=42,
                cache_root=str(tmp_path / "task-cache"),
                record_trials=0,
                asset_base_url=None,
                task_api_base_url=None,
                asset_fetch_workers=1,
                refresh_task=False,
                prepare_only=False,
                dry_run=False,
                host="localhost",
                port=0,
                max_control_steps=1,
                policy_seed=0,
                replan_steps=1,
                resize_size=224,
                result_path=str(tmp_path / "result.json"),
            )
        )
        assert rollout["status"] == "ok" and len(requests) == 2
        assert all(episode["official_randomization"]["visual"] for episode in rollout["episodes"])
    with pytest.raises(ValueError, match="already exists"):
        build(args)
    payload["checker_config"]["threshold"] = -1
    specs["checker_sha256"] = canonical_json_sha256(payload["checker_config"])
    source.write_text(json.dumps(manifest))
    (tmp_path / "source-tasks/1952.json").write_text(json.dumps(payload))
    binding["tasks"][0]["source_payload_sha256"] = canonical_json_sha256(payload)
    bindings.write_text(json.dumps(binding))
    args.output = str(tmp_path / "rejected-release")
    with pytest.raises(ValueError, match="already satisfies"):
        build(args)
    assert not pathlib.Path(args.output).exists()
