"""AXIS submissions use the evaluator's Pi0.5 inputs, not training sidecars."""

import json
import pathlib
import sys
import types
from unittest import mock

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))
from axis_model_input import model_uses_discrete_state
from axis_backend import _checkpoint_provenance
from check_model import check_model
from serve_axis_openpi import main as serve


def test_historical_training_provenance_remains_available_for_checkpoint_averaging():
    assert model_uses_discrete_state({}) is False
    assert model_uses_discrete_state({"discrete_state_input": False}) is False
    assert model_uses_discrete_state({"discrete_state_input": True}) is True
    for value in (0, 1, "false", "true", None):
        with pytest.raises(ValueError):
            model_uses_discrete_state({"discrete_state_input": value})


@pytest.mark.parametrize(
    "sidecar",
    [None, "{}", '{"discrete_state_input":false}', '{"discrete_state_input":true}', "{broken", "[]"],
)
def test_server_always_uses_standard_pi05_state_input(tmp_path, sidecar):
    if sidecar is not None:
        (tmp_path / "axis_vla_metadata.json").write_text(sidecar)
    make = mock.Mock(return_value=types.SimpleNamespace(model=types.SimpleNamespace(action_horizon=10, action_dim=32)))
    create_policy = mock.Mock(return_value=types.SimpleNamespace(metadata={}))
    websocket = mock.Mock()
    modules = {
        "axis_openpi_config": types.SimpleNamespace(make_config=make),
        "openpi.policies": types.SimpleNamespace(
            policy_config=types.SimpleNamespace(create_trained_policy=create_policy)
        ),
        "openpi.serving": types.SimpleNamespace(
            websocket_policy_server=types.SimpleNamespace(WebsocketPolicyServer=websocket)
        ),
    }
    with mock.patch.dict(sys.modules, modules), mock.patch(
        "axis_openpi_sources.bind_openpi_sources"
    ), mock.patch.object(sys, "argv", ["serve_axis_openpi.py", "--checkpoint", str(tmp_path)]):
        serve()
    make.assert_called_once_with(gripper_mode="continuous", discrete_state_input=True)
    assert create_policy.call_args.args[1] == tmp_path
    websocket.return_value.serve_forever.assert_called_once()


def test_standard_checkpoint_needs_norm_stats_but_not_internal_metadata(tmp_path):
    (tmp_path / "params").mkdir()
    stats_path = tmp_path / "assets/axis-v0.1-task501-runtime-v1/norm_stats.json"
    stats_path.parent.mkdir(parents=True)
    stats = {"mean": [0.0] * 9, "std": [1.0] * 9, "q01": [-1.0] * 9, "q99": [1.0] * 9}
    stats_path.write_text(json.dumps({"norm_stats": {"state": stats, "actions": stats}}))
    with mock.patch("check_model._check_jax_params"):
        assert check_model(tmp_path, "pi05_axis_joint").ok
        (tmp_path / "axis_vla_metadata.json").write_text("not a required submission file")
        assert check_model(tmp_path, "pi05_axis_joint").ok
        stats_path.unlink()
        result = check_model(tmp_path, "pi05_axis_joint")
        assert not result.ok
        assert any("missing normalization stats" in error for error in result.errors)


@pytest.mark.parametrize(
    "sidecar", [None, "{broken", "[]", '{"config":"pi05_axis_joint","discrete_state_input":"bad"}']
)
def test_optional_training_provenance_cannot_fail_score_summary(tmp_path, sidecar):
    if sidecar is not None:
        (tmp_path / "axis_vla_metadata.json").write_text(sidecar)
    provenance = _checkpoint_provenance(tmp_path)
    assert provenance is None or "discrete_state_input" not in provenance
