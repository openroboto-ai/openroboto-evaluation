"""Public replay export uses the released task bundle and preserves frame timing."""

import pathlib
import sys
import types
from unittest import mock

import numpy as np
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import export_axis_vla_dataset as exporter
from axis_vla import load_artifact


class SourceArray:
    """Zarr 3 provides shape and slicing, but does not implement len()."""

    def __init__(self, values):
        self.values = values
        self.shape = values.shape
        self.ndim = values.ndim

    def __getitem__(self, index):
        return self.values[index]


class ReplayEnvironment:
    def __init__(self, scene_path, payload, runtime):
        assert payload["id"] == 501
        assert runtime["camera"] == "camera0"
        self.closed = False

    def reset(self):
        self.state = np.zeros(9, dtype=np.float32)

    def render(self):
        return np.full((4, 4, 3), self.state[0], dtype=np.uint8)

    def observation_state(self):
        return self.state.copy()

    def step(self, action):
        self.state = action.copy()

    def success(self):
        passed = bool(self.state[0] >= 2)
        return passed, {"passed": passed}

    def close(self):
        self.closed = True


@pytest.fixture
def replay_source(tmp_path, monkeypatch):
    # First episode fails; second succeeds at its second sampled target.
    actions = np.zeros((24, 9), dtype=np.float32)
    actions[12] = 1
    actions[18] = 2
    group = {
        "data/state": SourceArray(np.zeros_like(actions)),
        "data/action": SourceArray(actions),
        "meta/episode_ends": np.asarray([12, 24]),
    }
    # Keep the scheduler's tests independent of the separate Zarr/MuJoCo install.
    # Frozen task loading and artifact serialization below use the real code.
    monkeypatch.setitem(sys.modules, "zarr", types.SimpleNamespace(open_group=lambda *a, **kw: group))
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    cache = mock.Mock()
    cache.prepare_scene.return_value = (tmp_path / "scene.xml", {})
    monkeypatch.setattr(exporter, "AssetCache", lambda *a, **kw: cache)
    environments = []

    def create_environment(*args):
        environment = ReplayEnvironment(*args)
        environments.append(environment)
        return environment

    monkeypatch.setattr(exporter, "AxisEnvironment", create_environment)
    output = tmp_path / "replays.npz"
    argv = [
        "--dataset",
        str(tmp_path / "source.zarr"),
        "--output",
        str(output),
        "--cache-root",
        str(tmp_path / "cache"),
    ]
    return group, argv, output, environments


def test_default_v1_export_filters_failures_and_captures_before_actions(replay_source):
    _, argv, output, environments = replay_source
    exporter.main(argv)
    artifact = load_artifact(output)
    assert artifact.metadata["benchmark"] == "axis_v1.0"
    assert artifact.metadata["source_frequency_hz"] == 30
    assert artifact.metadata["runtime_frequency_hz"] == 5
    assert artifact.metadata["source_episodes"] == [1]
    assert artifact.metadata["manifest_sha256"] == exporter.canonical_json_sha256(exporter.load_manifest())
    np.testing.assert_array_equal(artifact.states[:, 0], [0, 1])
    np.testing.assert_array_equal(artifact.images[:, 0, 0, 0], [0, 1])
    np.testing.assert_array_equal(artifact.actions[:, 0], [1, 2])
    np.testing.assert_array_equal(artifact.episode_ends, [2])
    assert environments[0].closed


def test_yaml_alias_uses_pinned_source_snapshot_directory(replay_source, tmp_path):
    _, argv, output, _ = replay_source
    # An edited YAML filename does not rename the source JSON or its snapshots.
    config = yaml.safe_load(exporter.DEFAULT_MANIFEST.read_text())
    config["tasks"] = [task for task in config["tasks"] if task["task_id"] == 501]
    manifest = tmp_path / "selected.yaml"
    manifest.write_text(yaml.safe_dump(config))
    source_path = exporter.DEFAULT_MANIFEST.parent / config["source_manifest"]
    manifest.with_name(source_path.name).write_bytes(source_path.read_bytes())
    snapshot_root = tmp_path / f"{source_path.stem}-tasks"
    snapshot_root.mkdir()
    (snapshot_root / "501.json").write_bytes((source_path.parent / snapshot_root.name / "501.json").read_bytes())
    exporter.main([*argv, "--manifest", str(manifest), "--episodes", "1"])
    assert load_artifact(output).metadata["source_episodes"] == [1]


def test_no_success_refuses_export_and_closes_runtime(replay_source):
    _, argv, output, environments = replay_source
    with pytest.raises(RuntimeError, match="no source episode passed"):
        exporter.main([*argv, "--episodes", "0"])
    assert not output.exists()
    assert environments[0].closed


@pytest.mark.parametrize("ends", [[], [0, 24], [12, 12, 24], [24, 12], [12, 25], [12.5, 24]])
def test_invalid_episode_boundaries_fail_before_runtime(replay_source, ends):
    group, argv, output, environments = replay_source
    group["meta/episode_ends"] = np.asarray(ends)
    with pytest.raises(ValueError, match="episode_ends"):
        exporter.main(argv)
    assert not output.exists()
    assert not environments


@pytest.mark.parametrize("hz,stride", [("nan", "6"), ("0", "6"), ("30", "0"), ("30", "5")])
def test_invalid_timing_fails_before_runtime(replay_source, hz, stride):
    _, argv, output, environments = replay_source
    with pytest.raises(SystemExit) as raised:
        exporter.main([*argv, "--source-frequency-hz", hz, "--stride", stride])
    assert raised.value.code == 2
    assert not output.exists()
    assert not environments


def test_explicit_source_rate_is_independent_of_manifest_provenance(replay_source):
    group, argv, output, _ = replay_source
    # Same target timing, with an input already sampled to the control rate.
    group["data/action"] = group["data/action"][::6]
    group["data/state"] = group["data/state"][::6]
    group["meta/episode_ends"] = np.asarray([2, 4])
    exporter.main([*argv, "--source-frequency-hz", "5", "--stride", "1"])
    artifact = load_artifact(output)
    assert artifact.metadata["source_frequency_hz"] == 5
    assert artifact.metadata["stride"] == 1
    np.testing.assert_array_equal(artifact.actions[:, 0], [1, 2])


def test_missing_source_arrays_have_actionable_error(replay_source):
    group, argv, _, environments = replay_source
    del group["data/action"]
    with pytest.raises(ValueError, match="source Zarr requires data/state, data/action"):
        exporter.main(argv)
    assert not environments


def test_missing_renderer_setting_fails_before_runtime(replay_source, monkeypatch):
    _, argv, _, environments = replay_source
    monkeypatch.delenv("MUJOCO_GL")
    with pytest.raises(ValueError, match="requires MUJOCO_GL='osmesa'"):
        exporter.main(argv)
    assert not environments
