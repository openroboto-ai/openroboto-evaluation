import json
import hashlib
import pathlib
import sys
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "libero_eval"), str(ROOT / "tools")]

from test_axis_scene import scene_fixture
from test_axis_randomization import TestAxisRandomization
from axis_backend import _summary
from axis_scene import prepare_scene, resolve_profile
from axis_randomization import PERMUTATION_ALGORITHM, build_trial_plan
from axis_runtime import AxisEnvironment, task_trial_count, task_runtime
from axis_runtime import canonical_json_sha256, load_manifest, task_specs
from axis_perturbations import has_domain_randomization
from build_axis_randomized_release import build
from test_axis_perturbations import fixture_environment, tiny_model
from plan_axis_randomized_release import matching_reset_source


def component_visual(payload, *, arena=True, wrist=False):
    visual = payload["official_randomization"]["visual"]
    visual.update(
        mode="official_franka_components",
        components={
            "arena": arena,
            "front_camera": True,
            "wrist_camera": wrist,
            "background": True,
            "surfaces": {"table": [], "floor": [], "wall": []} if arena else {},
        },
    )


@pytest.mark.parametrize("mode", ["official_franka_components", "official_franka_components_v2"])
def test_legacy_wrist_does_not_disable_front_materials_or_room(tmp_path, mode):
    scene, payload, runtime = scene_fixture(tmp_path)
    text = scene.read_text().replace('name="panda_link8"', 'name="panda_hand"')
    scene.write_text(text)
    component_visual(payload)
    payload["official_randomization"]["visual"]["mode"] = mode
    env = AxisEnvironment(scene, payload, runtime)
    try:
        env.reset()
        first = env.render()
        assert first.shape == (360, 640, 3)
        metadata = env.reset_randomization["visual"]
        assert set(metadata["camera_samples"]) == {"frontview"}
        assert set(metadata["material_samples"]) == {"table", "floor", "wall"}
        env.reset()
        np.testing.assert_array_equal(first, env.render())
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["official_franka_components", "official_franka_components_v2"])
def test_preserved_scene_does_not_shift_fixed_task_bodies(tmp_path, mode):
    scene, payload, runtime = scene_fixture(tmp_path)
    text = scene.read_text().replace(
        "</worldbody>", '<body name="cabinet" pos=".7 0 .3"><geom type="box" size=".1 .1 .3"/></body></worldbody>'
    )
    scene.write_text(text)
    component_visual(payload, arena=False, wrist=True)
    payload["official_randomization"]["visual"]["mode"] = mode
    env = AxisEnvironment(scene, payload, runtime)
    try:
        env.reset()
        visual = env.axis_scene
        # mj_step leaves derived transforms at the preceding integration state;
        # synchronize both models with the current qpos before comparing.
        env.mujoco.mj_forward(env.model, env.data)
        visual.sync(env.data)
        np.testing.assert_array_equal(visual.qpos_offset, 0)
        for name in ("cabinet", "franka/", "box"):
            np.testing.assert_array_equal(visual.data.body(name).xpos, env.data.body(name).xpos)
        assert visual.metadata["source_geometry_preserved"] is True
        assert env.render().shape == (360, 640, 3)
    finally:
        env.close()


def test_surface_only_does_not_require_table_placement(tmp_path):
    scene, payload, runtime = scene_fixture(tmp_path)
    # An out-of-footprint object must not prevent independently applying a
    # native surface material. It still prevents the full arena.
    scene.write_text(scene.read_text().replace('name="box" pos=".4 0 .02"', 'name="box" pos="5 0 .02"'))
    component_visual(payload, arena=False)
    visual = payload["official_randomization"]["visual"]
    visual["components"].update(front_camera=False, background=False)
    config, _ = resolve_profile(visual, payload["id"], payload["name"])
    _, metadata = prepare_scene(scene, config, visual["components"])
    assert metadata["source_geometry_preserved"]
    with pytest.raises(ValueError, match="footprint"):
        prepare_scene(scene, config)


def test_mixed_trials_keep_tasks_equally_weighted_and_errors_unscored():
    results = {
        1: {"status": "ok", "num_trials": 1, "num_successes": 1},
        2: {"status": "ok", "num_trials": 20, "num_successes": 0},
    }
    summary = _summary(results, {"benchmark": "axis_v1.1", "score_reduction": "task_mean"})
    assert summary["overall_success_rate"] == 0.5
    assert summary["episode_success_rate"] == 1 / 21
    assert task_trial_count({"randomization_enabled": False}, 20) == 1
    assert task_trial_count({"randomization_enabled": True}, 20) == 20
    results[2] = {"status": "error"}
    assert _summary(results, {"benchmark": "axis_v1.1", "score_reduction": "task_mean"})["overall_success_rate"] is None


def test_new_plan_cycles_all_variants_and_fixed_tasks_run_once(tmp_path):
    fixture = TestAxisRandomization()
    path, specs, raw = fixture._fixture(tmp_path)
    raw["schema_version"] = 2
    raw["seed_contract"]["algorithm"] = PERMUTATION_ALGORITHM
    raw["tasks"][0]["randomization_enabled"] = True
    path.write_text(json.dumps(raw))
    plan = fixture._load(path, specs)
    trials = build_trial_plan(plan, task_id=501, num_trials=6, seed=42)
    assert len({t.selection.spec.variant_id for t in trials[:3]}) == 3
    assert len({t.selection.spec.variant_id for t in trials[3:]}) == 3
    assert [t.selection for t in trials] == [
        t.selection for t in build_trial_plan(plan, task_id=501, num_trials=6, seed=42)
    ]
    raw["tasks"][0].update(randomization_enabled=False, variants=raw["tasks"][0]["variants"][:1])
    path.write_text(json.dumps(raw))
    plan = fixture._load(path, specs)
    assert not build_trial_plan(plan, task_id=501, num_trials=1, seed=42)[0].selection.provenance()["enabled"]
    with pytest.raises(ValueError, match="exactly one trial"):
        build_trial_plan(plan, task_id=501, num_trials=20, seed=42)


def test_runtime_overrides_cannot_change_checker_or_control_timing():
    manifest = {"runtime": {"camera": "camera0", "control_period_s": 0.2}}
    assert (
        task_runtime(manifest, {"runtime_overrides": {"camera": "frontview", "wrist_camera": None}})["camera"]
        == "frontview"
    )
    with pytest.raises(ValueError, match="only select cameras"):
        task_runtime(manifest, {"runtime_overrides": {"control_period_s": 1}})


def test_rare_swaps_are_random_even_if_two_probe_seeds_miss_them():
    model = tiny_model()
    assert has_domain_randomization(
        model,
        {"objects": {"box": {}, "ball": {}}, "swap_positions": {"pairs": [["box", "ball"]], "probability": 0.000001}},
    )
    assert not has_domain_randomization(model, {"objects": {"box": {"pos_delta": [[0.1, 0.1], [0, 0], [0, 0]]}}})


def test_builder_fixed_task_freezes_one_instance_and_one_trial(tmp_path, monkeypatch):
    _, payload, runtime = fixture_environment(tmp_path, visual=False)
    payload.pop("official_randomization")
    spec = dict(
        task_id=payload["id"],
        instruction=payload["name"],
        mjcf_sha256=hashlib.sha256(payload["mjcf_xml"].encode()).hexdigest(),
        checker_sha256=canonical_json_sha256(payload["checker_config"]),
        initial_state_sha256=canonical_json_sha256(payload["initial_state"]),
    )
    manifest = dict(
        name="axis_v90.0",
        status="runtime-ready",
        protocol_revision="base",
        protocol={"randomization": False},
        runtime=runtime,
        tasks=[spec],
    )
    source = tmp_path / "source.json"
    source.write_text(json.dumps(manifest))
    (tmp_path / "source-tasks").mkdir()
    (tmp_path / "source-tasks/1952.json").write_text(json.dumps(payload))
    binding = dict(
        schema_version=2,
        name="axis_v90.1",
        protocol_revision="mixed",
        namespace="mixed-test",
        tasks=[
            dict(
                task_id=1952,
                source_payload_sha256=canonical_json_sha256(payload),
                upstream_provenance={"test": "no components"},
                domain_randomization={},
                object_order=[],
                visual=None,
                randomization_enabled=False,
                runtime_overrides={},
            )
        ],
        instances=[
            dict(
                variant_id=f"official-{i:02d}",
                submit_nonce=f"{123456 + i:08d}",
                global_seed=42,
                attempt_id=0,
                render_variant_id=i,
                replica_id=i,
            )
            for i in range(2)
        ],
    )
    path = tmp_path / "bindings.json"
    path.write_text(json.dumps(binding))
    result = build(
        SimpleNamespace(
            source_manifest=source, bindings=path, output=tmp_path / "release", cache_root=tmp_path / "cache", workers=2
        )
    )
    assert result["instances"] == 1
    release = load_manifest(tmp_path / "release/benchmark.json")
    from axis_randomization import load_randomization_plan

    plan = load_randomization_plan(
        tmp_path / "release/randomization.json",
        expected_benchmark="axis_v90.1",
        expected_protocol_revision="mixed",
        benchmark_task_specs=task_specs(release),
    )
    assert not build_trial_plan(plan, task_id=1952, num_trials=1, seed=42)[0].selection.enabled
    assert release["protocol"]["score_reduction"] == "task_mean"
    assert task_trial_count(task_specs(release)[1952], 20) == 1
    import axis_task

    requests = []

    class Policy:
        def infer(self, request):
            requests.append(request)
            return {"actions": request["observation/state"][None, :]}

    monkeypatch.setattr(axis_task, "_policy_client", lambda host, port: Policy())
    monkeypatch.setattr(axis_task, "_resize_image", lambda image, size: image)
    for prepare_only, dry_run in [(True, False), (False, True), (False, False)]:
        args = SimpleNamespace(
            manifest=str(tmp_path / "release/benchmark.json"),
            task_id=1952,
            num_trials=20,
            randomization_manifest=str(tmp_path / "release/randomization.json"),
            randomization_seed=42,
            cache_root=str(tmp_path / "task-cache"),
            asset_base_url=None,
            task_api_base_url=None,
            asset_fetch_workers=1,
            refresh_task=False,
            record_trials=0,
            prepare_only=prepare_only,
            dry_run=dry_run,
            smoke_control_steps=1,
            smoke_render_frames=1,
            host="localhost",
            port=0,
            max_control_steps=1,
            policy_seed=0,
            replan_steps=1,
            resize_size=224,
            result_path=str(tmp_path / "result.json"),
        )
        result = axis_task.run(args)
        assert result["status"] == "ok"
        assert result["randomization"] is False
        assert len(result["trial_variants"]) == 1
        assert result["trial_variants"][0]["enabled"] is False
    assert result["num_trials"] == 1
    assert len(requests) == 1
    payload["checker_config"]["threshold"] = -1
    spec["checker_sha256"] = canonical_json_sha256(payload["checker_config"])
    source.write_text(json.dumps(manifest))
    (tmp_path / "source-tasks/1952.json").write_text(json.dumps(payload))
    binding["tasks"][0]["source_payload_sha256"] = canonical_json_sha256(payload)
    path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="already satisfies"):
        build(
            SimpleNamespace(
                source_manifest=source,
                bindings=path,
                output=tmp_path / "invalid-fixed",
                cache_root=tmp_path / "cache",
                workers=2,
            )
        )
    assert not (tmp_path / "invalid-fixed").exists()


def test_parallel_planner_records_each_native_probe(tmp_path):
    _, payload, runtime = fixture_environment(tmp_path, visual=False)
    payload.pop("official_randomization")
    snapshots = tmp_path / "source-tasks"
    snapshots.mkdir()
    specs = []
    for tid in (1952, 1953):
        task = {**payload, "id": tid}
        (snapshots / f"{tid}.json").write_text(json.dumps(task))
        specs.append(
            dict(
                task_id=tid,
                instruction=task["name"],
                mjcf_sha256=hashlib.sha256(task["mjcf_xml"].encode()).hexdigest(),
                checker_sha256=canonical_json_sha256(task["checker_config"]),
                initial_state_sha256=canonical_json_sha256(task["initial_state"]),
            )
        )
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            dict(
                name="axis_v90.0",
                status="runtime-ready",
                protocol_revision="base",
                protocol={"randomization": False},
                runtime=runtime,
                tasks=specs,
            )
        )
    )
    output = tmp_path / "bindings.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/plan_axis_randomized_release.py"),
            "--source-manifest",
            str(source),
            "--cache-root",
            str(tmp_path / "cache"),
            "--name",
            "axis_v90.1",
            "--output",
            str(output),
            "--workers",
            "2",
        ],
        env={**os.environ, "MUJOCO_GL": "osmesa", "LP_NUM_THREADS": "2"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    bindings = json.loads(output.read_bytes())
    assert bindings["protocol_revision"] == "axis_v90.1_official_components_native_mujoco_v2"
    assert {b["task_id"] for b in bindings["tasks"]} == {1952, 1953}
    for binding in bindings["tasks"]:
        assert json.loads((tmp_path / "bindings-tasks" / f"{binding['task_id']}.json").read_bytes()) == binding
        assert binding["visual"]["components"]["front_camera"] is True
        assert binding["visual"]["mode"] == "official_franka_components_v2"


def test_robot_home_pose_does_not_hide_a_matching_official_object_reset():
    source = dict(
        id=1,
        mjcf_xml="frozen XML",
        checker_config={"type": "frozen checker"},
        initial_state=None,
        domain_randomization=None,
    )
    frozen = {**source, "initial_state": {"robots": {"franka": {"dof_pos": {"panda_joint1": 0.2}}}}}
    assert matching_reset_source(frozen, source)
    # Object/base positions and task/checker revisions still require an exact
    # match. This exception is only for the nine policy-controlled joint values.
    for change in [
        dict(initial_state={"objects": {"box": {"pos": [0.1, 0, 0]}}}),
        dict(initial_state={"robots": {"franka": {"pos": [0.1, 0, 0]}}}),
        dict(mjcf_xml="new scene"),
        dict(checker_config={"type": "different checker"}),
        dict(id=2),
    ]:
        assert not matching_reset_source({**frozen, **change}, source)


@pytest.mark.parametrize("source_height", [0.0, 0.48])
def test_partial_background_and_camera_metadata_use_source_frame(source_height, tmp_path):
    scene, payload, runtime = scene_fixture(tmp_path)
    scene.write_text(
        scene.read_text().replace('name="franka/" pos="0 0 0"', f'name="franka/" pos="0 0 {source_height}"')
    )
    visual = payload["official_randomization"]["visual"]
    visual.update(
        mode="official_franka_components_v2",
        components=dict(arena=False, background=True, front_camera=True, wrist_camera=False, surfaces={}),
    )
    config, _ = resolve_profile(visual, payload["id"], payload["name"])
    _, old_meta = prepare_scene(scene, config, visual["components"])
    xml, new_meta = prepare_scene(scene, config, visual["components"], source_frame=True)
    assert old_meta["table_top_z"] == 0.48  # frozen V1 behavior remains reproducible
    assert new_meta["table_top_z"] == source_height
    assert new_meta["source_frame_z_translation"] == pytest.approx(source_height - 0.48)
    for old, new in zip(old_meta["fixtures"], new_meta["fixtures"]):
        np.testing.assert_allclose(
            np.array(new["absolute_center_pos"]) - old["absolute_center_pos"], [0, 0, source_height - 0.48], atol=1e-12
        )
    env = AxisEnvironment(scene, payload, runtime)
    try:
        env.reset()
        sample = env.reset_randomization["visual"]["camera_samples"]["frontview"]
        assert env.reset_randomization["visual"]["xml_metadata"]["table_top_z"] == source_height
        assert np.isfinite(env.render()).all()
        assert sample["geometry"]["table_world_corners"][0][2] == source_height
    finally:
        env.close()
