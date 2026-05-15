import pytest

pytest.skip(
    "ported from safe_sim2real under OpenSO-101 skeleton mode; "
    "awaits implementation port. See "
    "/data/safe_sim2real/tests/test_lerobot_so101_interface.py for the legacy assertions.",
    allow_module_level=True,
)

import math

from openso101.teleop.so101_mapping import SO101_TELEOP_TARGET_LIMITS_DEG


class FakeLeaderRobot:
    def __init__(self):
        self.connected = False

    def connect(self):
        self.connected = True

    def get_action(self):
        return {
            "shoulder_pan.pos": 100.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 50.0,
        }


def test_leader_wrapper_connects_injected_robot_and_maps_targets():
    from openso101.teleop.lerobot_interface import LeRobotSO101Leader

    fake_robot = FakeLeaderRobot()
    leader = LeRobotSO101Leader(port="/dev/null", robot_id="test_leader", robot=fake_robot)

    leader.connect()
    action, targets = leader.read_ordered_targets()

    assert fake_robot.connected is True
    assert action["shoulder_pan.pos"] == 100.0
    assert targets[0] == pytest.approx(math.radians(SO101_TELEOP_TARGET_LIMITS_DEG["shoulder_pan"].upper))
