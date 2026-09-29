import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from test_axis_perturbations import fixture_environment
from audit_axis_resets import audit_task, validated_bindings
from axis_runtime import canonical_json_sha256


@pytest.mark.parametrize(
    "initial_x, delta, expected",
    [(0.4, 0.01, "retain"), (0.4, 0.15, "disable_physical_reset"), (0.55, 0.01, "blocked")],
)
def test_reset_audit_distinguishes_bad_perturbation_from_bad_base(tmp_path, initial_x, delta, expected):
    _, payload, runtime = fixture_environment(tmp_path, visual=False)
    payload.pop("official_randomization")
    payload["mjcf_xml"] = payload["mjcf_xml"].replace('name="box" pos=".4', f'name="box" pos="{initial_x}')
    payload["checker_config"] = {
        "type": "RelativePositionBoundsChecker",
        "objName": "box",
        "refName": "franka/",
        "xRange": [0.5, 0.6],
    }
    binding = {
        "task_id": 1952,
        "source_payload_sha256": canonical_json_sha256(payload),
        "upstream_provenance": {"enabled_components": {"physical_reset": True}, "unavailable_components": {}},
        "domain_randomization": {"objects": {"box": {"pos_delta": [[delta, delta], [0, 0], [0, 0]]}}},
        "object_order": ["box"],
        "randomization_enabled": True,
        "visual": None,
    }
    instances = [{"variant_id": "official-00", "submit_nonce": "00123456"}]
    result = audit_task(({"task_id": 1952}, payload, binding, instances, runtime, tmp_path / "cache", tmp_path))
    assert result["recommendation"] == expected
    assert result["base"]["valid"] is (expected != "blocked")
    assert result["trials"]["official-00"]["valid"] is (expected == "retain")
    assert json.loads((tmp_path / "1952.json").read_text()) == result
    assert "official_randomization" not in payload
    assert instances == [{"variant_id": "official-00", "submit_nonce": "00123456"}]
    bindings = {"tasks": [binding], "instances": instances}
    report = {"tasks": [result], "bindings_sha256": canonical_json_sha256(bindings)}
    if expected == "blocked":
        with pytest.raises(ValueError, match="no valid audited base"):
            validated_bindings(bindings, report)
    else:
        adjusted = validated_bindings(bindings, report)
        assert adjusted["instances"] == instances
        assert binding["domain_randomization"]  # Source bindings remain unchanged.
        assert adjusted["tasks"][0]["randomization_enabled"] is (expected == "retain")
        assert bool(adjusted["tasks"][0]["domain_randomization"]) is (expected == "retain")
    with pytest.raises(ValueError, match="different bindings"):
        validated_bindings({**bindings, "instances": []}, report)
