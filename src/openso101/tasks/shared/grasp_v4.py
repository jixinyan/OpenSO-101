from openso101.robots import SO101_ARM_JOINT_NAMES, SO101_GRIPPER_JOINT_NAMES

from .grasp_v3 import configure_grasp_v3
from .position_action import NormalizedJointPositionActionCfg


def configure_grasp_v4(cfg, task_id):
    configure_grasp_v3(cfg, task_id)
    cfg.scene.robot.soft_joint_pos_limit_factor = 1.0
    cfg.actions.arm_action = NormalizedJointPositionActionCfg(
        asset_name="robot", joint_names=list(SO101_ARM_JOINT_NAMES), preserve_order=True,
    )
    cfg.actions.gripper_action = NormalizedJointPositionActionCfg(
        asset_name="robot", joint_names=list(SO101_GRIPPER_JOINT_NAMES), preserve_order=True,
        clip={".*": (0., .8)},
    )
