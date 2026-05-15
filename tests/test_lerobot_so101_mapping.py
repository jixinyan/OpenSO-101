import pytest

pytest.skip(
    "ported from safe_sim2real under OpenSO-101 skeleton mode; "
    "awaits implementation port. See "
    "/data/safe_sim2real/tests/test_lerobot_so101_mapping.py for the legacy assertions.",
    allow_module_level=True,
)

import math

from openso101.robots import SO101_SIM_JOINT_NAMES
from openso101.teleop.so101_mapping import (
    LEROBOT_SO101_ACTION_NAMES,
    SO101_TELEOP_CONTROL_JOINT_NAMES,
    SO101_TELEOP_TARGET_LIMITS_DEG,
    lerobot_action_to_joint_targets,
    lerobot_action_to_ordered_targets,
    get_sim_joint_names,
    parse_joint_name_set,
    parse_joint_offsets_deg,
    transform_ordered_targets,
)


def _leader_action(**overrides):
    action = {name: 0.0 for name in LEROBOT_SO101_ACTION_NAMES}
    action["gripper.pos"] = 50.0
    action.update(overrides)
    return action


def test_midpoint_leader_action_maps_to_sim_joint_midpoints():
    targets = lerobot_action_to_joint_targets(_leader_action())

    for joint_name, limits in SO101_TELEOP_TARGET_LIMITS_DEG.items():
        expected_deg = (limits.lower + limits.upper) / 2.0
        assert targets[joint_name] == pytest.approx(math.radians(expected_deg))


def test_endpoint_leader_action_maps_to_sim_joint_limits():
    action = _leader_action(
        **{
            "shoulder_pan.pos": 100.0,
            "shoulder_lift.pos": -100.0,
            "elbow_flex.pos": 100.0,
            "wrist_flex.pos": -100.0,
            "wrist_roll.pos": 100.0,
            "gripper.pos": 100.0,
        }
    )

    targets = lerobot_action_to_joint_targets(action)

    assert targets["shoulder_pan"] == pytest.approx(math.radians(SO101_TELEOP_TARGET_LIMITS_DEG["shoulder_pan"].upper))
    assert targets["shoulder_lift"] == pytest.approx(math.radians(SO101_TELEOP_TARGET_LIMITS_DEG["shoulder_lift"].lower))
    assert targets["elbow_flex"] == pytest.approx(math.radians(SO101_TELEOP_TARGET_LIMITS_DEG["elbow_flex"].upper))
    assert targets["wrist_flex"] == pytest.approx(math.radians(SO101_TELEOP_TARGET_LIMITS_DEG["wrist_flex"].lower))
    assert targets["wrist_roll"] == pytest.approx(math.radians(SO101_TELEOP_TARGET_LIMITS_DEG["wrist_roll"].upper))
    assert targets["gripper"] == pytest.approx(math.radians(SO101_TELEOP_TARGET_LIMITS_DEG["gripper"].upper))


def test_gripper_uses_canonical_usd_jaw_range():
    closed = lerobot_action_to_joint_targets(_leader_action(**{"gripper.pos": 0.0}))
    opened = lerobot_action_to_joint_targets(_leader_action(**{"gripper.pos": 100.0}))

    assert closed["gripper"] == pytest.approx(math.radians(-10.0))
    assert opened["gripper"] == pytest.approx(math.radians(100.0))


def test_ordered_targets_match_canonical_so101_action_order():
    ordered = lerobot_action_to_ordered_targets(_leader_action(**{"shoulder_pan.pos": 100.0}))

    assert len(ordered) == 6
    assert get_sim_joint_names() == SO101_SIM_JOINT_NAMES
    assert ordered[0] == pytest.approx(math.radians(SO101_TELEOP_TARGET_LIMITS_DEG["shoulder_pan"].upper))


def test_missing_leader_joint_reports_action_name():
    action = _leader_action()
    del action["wrist_roll.pos"]

    with pytest.raises(KeyError, match="wrist_roll\\.pos"):
        lerobot_action_to_joint_targets(action)


def test_transform_ordered_targets_applies_inversion_offsets_and_clamps():
    targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    transformed = transform_ordered_targets(
        targets,
        inverted_joints={"shoulder_lift"},
        offsets_rad={"elbow_flex": 0.1, "gripper": 2.0},
    )

    assert transformed[0] == pytest.approx(0.1)
    assert transformed[1] == pytest.approx(-0.2)
    assert transformed[2] == pytest.approx(0.4)
    assert transformed[5] == pytest.approx(math.radians(SO101_TELEOP_TARGET_LIMITS_DEG["gripper"].upper))


def test_parse_joint_calibration_options():
    assert parse_joint_name_set("shoulder_lift, wrist_flex") == frozenset({"shoulder_lift", "wrist_flex"})
    assert parse_joint_offsets_deg("elbow_flex:10,gripper:-5") == {
        "elbow_flex": pytest.approx(math.radians(10)),
        "gripper": pytest.approx(math.radians(-5)),
    }

    with pytest.raises(ValueError, match="bad_joint"):
        parse_joint_name_set("bad_joint")


def test_control_joint_names_are_the_logical_lerobot_order():
    assert SO101_TELEOP_CONTROL_JOINT_NAMES == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
