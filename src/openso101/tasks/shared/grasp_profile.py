import torch
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from openso101.robots import SO101_GRIPPER_JOINT_NAMES
from openso101.tasks.shared.grasp import object_grasped_by_jaws


def grasp_alignment(env):
    robot = env.scene["robot"]
    index = robot.body_names.index("gripper")
    local_position, _ = subtract_frame_transforms(
        robot.data.body_pos_w[:, index], robot.data.body_quat_w[:, index],
        env.scene["object"].data.root_pos_w,
    )
    # 使用实际 ee_frame 定义的抓取中心，分别计算三个方向的距离。
    center = torch.tensor((0.01, 0.0, -0.09), device=env.device)
    extent = torch.tensor((0.04, 0.025, 0.03), device=env.device)
    return torch.exp(-torch.sum(((local_position - center) / extent).square(), dim=-1))


def approach(env):
    distance = torch.linalg.vector_norm(
        env.scene["object"].data.root_pos_w - env.scene["ee_frame"].data.target_pos_w[:, 0], dim=-1,
    )
    return (1 - torch.tanh(distance / 0.2)) * ~object_grasped_by_jaws(env)


def closure_at_object(env):
    target = env.action_manager.get_term("gripper_action").processed_actions[:, 0]
    closed = (1 - target / 0.8).clamp(0, 1)
    return grasp_alignment(env) * closed * ~release_ready(env)


def closure_away_from_object(env):
    target = env.action_manager.get_term("gripper_action").processed_actions[:, 0]
    return (1 - grasp_alignment(env)) * (1 - target / 0.8).clamp(0, 1) * ~object_grasped_by_jaws(env)


def contact_hold(env):
    return object_grasped_by_jaws(env).float()


def held_height(env):
    height = env.scene["object"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return object_grasped_by_jaws(env) * ((height - 0.015) / 0.10).clamp(0, 1)


def goal_distance(env):
    robot = env.scene["robot"]
    goal, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        env.command_manager.get_command("object_pose")[:, :3],
    )
    return torch.linalg.vector_norm(env.scene["object"].data.root_pos_w - goal, dim=-1)


def held_goal(env):
    return object_grasped_by_jaws(env) * (1 - torch.tanh(goal_distance(env) / 0.10))


def release_ready(env):
    if env.cfg.task_profile_task == "lift":
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    command = env.command_manager.get_term("object_pose")
    # 最终放置阶段进入 3cm 目标范围后提供释放指令奖励。
    return (command.stage == 2) & (goal_distance(env) <= 0.03)


def release_at_goal(env):
    target = env.action_manager.get_term("gripper_action").processed_actions[:, 0]
    return release_ready(env) * (target / 0.8).clamp(0, 1)


def stable_placement(env):
    command = env.command_manager.get_term("object_pose")
    return (command.placement_hold_seconds / 0.5).clamp(0, 1)


def processed_action_change(env):
    current = torch.cat([env.action_manager.get_term(name).processed_actions
                         for name in env.action_manager.active_terms], dim=-1)
    previous = env.extras.get("_grasp_v2_targets")
    penalty = torch.zeros(env.num_envs, device=env.device)
    if previous is not None:
        penalty = (current - previous).square().sum(dim=-1)
        penalty[env.episode_length_buf <= 1] = 0
    env.extras["_grasp_v2_targets"] = current.detach().clone()
    return penalty


class HeldLiftSuccess(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.hold_seconds = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.hold_seconds[env_ids if env_ids is not None else slice(None)] = 0

    def __call__(self, env, minimal_height=0.04, goal_radius=0.05,
                 command_name="object_pose", settle_seconds=0.25, force_threshold=0.5):
        height = env.scene["object"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        eligible = (height > minimal_height) & (goal_distance(env) < goal_radius)
        eligible &= object_grasped_by_jaws(env, force_threshold)
        self.hold_seconds = torch.where(eligible, self.hold_seconds + env.step_dt, 0.)
        return self.hold_seconds >= settle_seconds


@configclass
class GraspRewardsCfg:
    approach = RewardTermCfg(func=approach, weight=1.)
    alignment = RewardTermCfg(func=grasp_alignment, weight=1.)
    closure = RewardTermCfg(func=closure_at_object, weight=2.)
    closure_away = RewardTermCfg(func=closure_away_from_object, weight=-0.5)
    grasp_hold = RewardTermCfg(func=contact_hold, weight=8.)
    held_height = RewardTermCfg(func=held_height, weight=12.)
    held_goal = RewardTermCfg(func=held_goal, weight=16.)
    processed_action_change = RewardTermCfg(func=processed_action_change, weight=-0.1)
    joint_vel = RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-4)
    success_bonus = RewardTermCfg(func=mdp.is_terminated_term, params={"term_keys": ["success"]}, weight=100.)


def configure_grasp_profile(cfg, task_id):
    if task_id not in ("OpenSO101-Lift-v0", "OpenSO101-PickPlace-v0"):
        raise ValueError("grasp_v2 仅支持 Lift 和 PickPlace")
    cfg.task_profile_task = "lift" if task_id == "OpenSO101-Lift-v0" else "pick_place"
    cfg.actions.gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=list(SO101_GRIPPER_JOINT_NAMES),
        scale=0.4, offset=0.4, use_default_offset=False,
        clip={".*": (0., 0.8)},
    )
    cfg.rewards = GraspRewardsCfg()
    cfg.curriculum = None
    if cfg.task_profile_task == "lift":
        cfg.terminations.success = TerminationTermCfg(
            func=HeldLiftSuccess,
            params={"minimal_height": 0.04, "goal_radius": 0.05,
                    "command_name": "object_pose", "settle_seconds": 0.25, "force_threshold": 0.5},
        )
    else:
        cfg.rewards.release = RewardTermCfg(func=release_at_goal, weight=12.)
        cfg.rewards.stable_placement = RewardTermCfg(func=stable_placement, weight=20.)
