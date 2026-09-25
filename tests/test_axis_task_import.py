import json
import pathlib
import sys
import xml.etree.ElementTree as ET

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from import_axis_tasks import normalize_checker, normalize_task  # noqa: E402


def test_pinned_import_preserves_objects_and_makes_viewer_ready_pose_explicit():
    source = json.loads((ROOT / "configs/axis-v1.0/tasks_config.json").read_text())
    task = next(task for task in source["tasks"] if task["id"] == 35)
    before = json.loads(task["initial_state"])
    result, changes = normalize_task(task)
    assert result["initial_state"]["objects"] == before["objects"]
    assert result["initial_state"]["robots"]["franka"]["dof_pos"]["panda_joint4"] == -2.356194
    assert 'name="camera0"' in result["mjcf_xml"]
    assert any("camera" in change for change in changes)
    assert json.loads(task["initial_state"]) == before


def test_legacy_alias_conflicts_are_rejected_and_recursive_aliases_normalized():
    result = normalize_checker({"checker": {"type": "RelativeCylinderChecker", "obj_name": "pot/", "ref_type": "site"}})
    assert result["checker"]["objName"] == "pot/"
    assert result["checker"]["refType"] == "site"
    with pytest.raises(ValueError, match="conflicting"):
        normalize_checker({"obj_name": "pot/", "objName": "bowl/"})


def test_import_cannot_turn_absent_checker_into_a_task():
    with pytest.raises(ValueError, match="no checker"):
        normalize_task({"id": 1, "checker_config": None})


@pytest.mark.parametrize("task_id,joint", [(49, "bottom"), (55, "top")])
def test_reviewed_drawer_index_bug_is_corrected_explicitly_without_mutating_source(task_id, joint):
    source = json.loads((ROOT / "configs/axis-v1.0/tasks_config.json").read_text())
    task = next(task for task in source["tasks"] if task["id"] == task_id)
    before = json.dumps(task, sort_keys=True)
    result, changes = normalize_task(task)
    root = result["checker_config"]["checker"]
    assert root["jointName"] == f"white_cabinet/{joint}_level"
    assert "jointIndex" not in root
    assert any("Benchmark predicate correction" in change for change in changes)
    assert json.dumps(task, sort_keys=True) == before


def test_stove_ground_intersection_is_corrected_and_audited():
    source = json.loads((ROOT / "configs/axis-v1.0/tasks_config.json").read_text())
    task = next(task for task in source["tasks"] if task["id"] == 44)
    payload, changes = normalize_task(task)
    stove = next(body for body in ET.fromstring(payload["mjcf_xml"]).iter("body") if body.get("name") == "flat_stove/")
    assert float(stove.get("pos").split()[2]) == 0
    assert any("7 mm" in change and "singular contact" in change for change in changes)
    assert payload["checker_config"] == json.loads(task["checker_config"])
