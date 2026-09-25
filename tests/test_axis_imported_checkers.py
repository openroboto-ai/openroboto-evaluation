"""Geometric/boundary regressions for the historical AXIS predicates."""

import math
import pathlib
import sys

import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "libero_eval"))
from axis_runtime import RuntimeState, evaluate_checker  # noqa: E402


def test_frame_box_rotates_into_site_frame_and_keeps_namespaces_separate():
    state = RuntimeState(
        {"object": [1, 2.2, 3], "frame": [99, 99, 99]},
        {},
        {},
        {"frame": [1, 2, 3]},
        {"frame": [0, 0, math.sqrt(0.5), math.sqrt(0.5)]},
    )
    checker = {
        "type": "FrameBBoxChecker",
        "objName": "object",
        "frameName": "frame",
        "frameType": "site",
        "lower": [0.19, -0.01, -0.01],
        "upper": [0.21, 0.01, 0.01],
    }
    assert evaluate_checker(checker, state, state)[0]
    checker["frameType"] = "object"
    assert not evaluate_checker(checker, state, state)[0]


def test_drawer_box_tracks_joint_and_relative_wxyz_rotation():
    state = RuntimeState(
        {"object": [0, -0.3, 0.18], "cabinet/": [0, 0, 0]},
        {"cabinet/": [0, 0, 0, 1]},
        {"cabinet/bottom": 0, "cabinet/top": -0.3},
    )
    checker = {
        "type": "DrawerBBoxChecker",
        "objName": "object",
        "cabinetName": "cabinet/",
        "jointIndex": 1,
        "baseOffset": [0, 0, 0],
        "halfSize": [0.2, 0.05, 0.03],
        "relativeQuat": [math.sqrt(0.5), 0, math.sqrt(0.5), 0],
    }
    assert evaluate_checker(checker, state, state)[0]
    state.joints["cabinet/top"] = 0
    assert not evaluate_checker(checker, state, state)[0]
    checker["jointIndex"] = 5
    assert not evaluate_checker(checker, state, state)[0]


@pytest.mark.parametrize("mode,expected", [("gt", False), ("ge", True), ("lt", False), ("le", True)])
def test_joint_threshold_exact_boundary(mode, expected):
    state = RuntimeState({}, {}, {"cabinet/top": -0.1})
    checker = {
        "type": "JointThresholdChecker",
        "objName": "cabinet/",
        "jointName": "top",
        "mode": mode,
        "threshold": -0.1,
    }
    assert evaluate_checker(checker, state, state)[0] is expected


def test_sample_delta_uses_configured_reference_and_delta_keys():
    state = RuntimeState({"sample/": [0.7, 9, 0.3]}, {}, {})
    checker = {
        "type": "SamplePositionDeltaChecker",
        "initialPosition": [0.55, 0, 0.2],
        "minDeltaX": 0.08,
        "minDeltaZ": 0.06,
    }
    assert evaluate_checker(checker, state, state)[0]
    state.positions["sample/"][2] = 0.25
    assert not evaluate_checker(checker, state, state)[0]
    with pytest.raises(ValueError, match="bound"):
        evaluate_checker({"type": "SamplePositionDeltaChecker"}, state, state)


def test_directional_tilt_ignores_yaw_and_rejects_opposite_direction():
    initial = RuntimeState({}, {"sample/": [0, 0, 0, 1]}, {})
    checker = {"type": "SampleRotationChecker", "tiltWorldDirection": [1, 0, 0], "tipAngleThreshold": 20}
    current = RuntimeState({}, {"sample/": [0, math.sin(math.pi / 6), 0, math.cos(math.pi / 6)]}, {})
    assert evaluate_checker(checker, current, initial)[0]
    current.orientations["sample/"][1] *= -1
    assert not evaluate_checker(checker, current, initial)[0]
    current.orientations["sample/"] = [0, 0, 1, 0]
    assert not evaluate_checker(checker, current, initial)[0]


def test_relative_cylinder_resolves_site_without_matching_body():
    state = RuntimeState({"pot": [0, 0, 0.04]}, {}, {}, {"burner": [0, 0, 0]})
    checker = {
        "type": "RelativeCylinderChecker",
        "objName": "pot",
        "refName": "burner",
        "refType": "site",
        "heightMax": 0.08,
    }
    assert evaluate_checker(checker, state, state)[0]
    checker["refType"] = "object"
    assert not evaluate_checker(checker, state, state)[0]


def test_bowl_bounds_and_invalid_no_constraint_fail_closed():
    state = RuntimeState({"bowl/": [0.5, 0.2, 0.3]}, {}, {})
    checker = {"type": "BowlPositionChecker", "minBounds": [0.4, 0.1, 0.2], "maxBounds": [0.6, 0.3, 0.4]}
    assert evaluate_checker(checker, state, state)[0]
    with pytest.raises(ValueError, match="requires"):
        evaluate_checker({"type": "BowlPositionChecker"}, state, state)
