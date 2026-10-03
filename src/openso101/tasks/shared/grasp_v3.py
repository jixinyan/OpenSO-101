import torch
from isaaclab.managers import ObservationTermCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply_inverse

from openso101.robots import SO101_ARM_JOINT_NAMES, SO101_GRIPPER_JOINT_NAMES
from .delta_action import JointDeltaActionCfg
from .grasp_profile import configure_grasp_profile
from .progress_reward import ProgressRewardsCfg


def object_velocity(env):
    robot, obj = env.scene["robot"], env.scene["object"]
    return torch.cat((quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w),
                      quat_apply_inverse(robot.data.root_quat_w, obj.data.root_ang_vel_w)), dim=-1)


def task_state(env):
    remaining = (1 - env.episode_length_buf.float() / env.max_episode_length).clamp(0, 1)
    if env.cfg.task_profile_task == "pick_place":
        command = env.command_manager.get_term("object_pose")
        return torch.stack((command.stage.float() / 2., command.placement_hold_seconds / .5, remaining), dim=-1)
    # ObservationManager 在 TerminationManager 之前创建。
    if not hasattr(env, "termination_manager"):
        return torch.stack((torch.zeros_like(remaining), torch.zeros_like(remaining), remaining), dim=-1)
    hold = env.termination_manager.get_term_cfg("success").func.hold_seconds
    return torch.stack((torch.zeros_like(hold), hold / .25, remaining), dim=-1)


def configure_grasp_v3(cfg, task_id):
    configure_grasp_profile(cfg, task_id)
    cfg.scene.robot.init_state.joint_pos["Wrist_Pitch"] = 1.4
    cfg.sim.dt = .002
    cfg.decimation = 10
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physx.solve_articulation_contact_last = True
    cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 8
    for asset in (cfg.scene.robot, cfg.scene.object):
        asset.spawn.rigid_props.max_depenetration_velocity = .1
    cfg.scene.object.spawn.rigid_props.solver_velocity_iteration_count = 8
    for actuator in cfg.scene.robot.actuators.values():
        actuator.armature = .028
        actuator.stiffness = 17.8
        actuator.damping = .6
        actuator.effort_limit_sim = 3.35
    cfg.reward_discount = .99
    cfg.terminations.time_out.time_out = False
    if cfg.task_profile_task == "pick_place":
        from openso101.tasks.pick_place.mdp.terminations import StablePlacementSuccess

        cfg.current_step_placement_success = True
        cfg.terminations.success = TerminationTermCfg(func=StablePlacementSuccess, params={"settle_seconds": .5})
    step_dt = cfg.sim.dt * cfg.decimation
    cfg.actions.arm_action = JointDeltaActionCfg(
        asset_name="robot", joint_names=list(SO101_ARM_JOINT_NAMES),
        preserve_order=True, scale=2. * step_dt,
    )
    cfg.actions.gripper_action = JointDeltaActionCfg(
        asset_name="robot", joint_names=list(SO101_GRIPPER_JOINT_NAMES),
        preserve_order=True, scale=2. * step_dt, clip={".*": (0., .8)},
    )
    cfg.observations.policy.object_velocity = ObservationTermCfg(func=object_velocity)
    cfg.observations.policy.task_state = ObservationTermCfg(func=task_state)
    cfg.rewards = ProgressRewardsCfg()
    cfg.rewards.success_bonus.weight = 30. / step_dt
    cfg.scene.ee_frame.debug_vis = False
    cfg.commands.object_pose.debug_vis = False


def configure_environment_mode(cfg, mode):
    if mode not in ("nominal", "randomized"):
        raise ValueError("environment_mode 需要 nominal 或 randomized")
    if mode == "nominal":
        for name in vars(cfg.events):
            if name.startswith("dr_") or name == "camera_mounts":
                setattr(cfg.events, name, None)
        cfg.action_dr_enabled = False
        cfg.observations.policy.enable_corruption = False
    cfg.environment_mode = mode
