import torch
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils import configclass

from .grasp import object_grasped_by_jaws
from .grasp_profile import grasp_alignment, goal_distance, processed_action_change, success_event


def task_potential(env):
    obj = env.scene["object"]
    distance = torch.linalg.vector_norm(obj.data.root_pos_w - env.scene["ee_frame"].data.target_pos_w[:, 0], dim=-1)
    grasped = object_grasped_by_jaws(env).float()
    height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    aligned = grasp_alignment(env)
    jaw = env.scene["robot"].data.joint_pos[:, env.scene["robot"].joint_names.index("Jaw")]
    approaching = (1 - torch.tanh(distance / .1)) + aligned
    closing = aligned * (1 - jaw / .8).clamp(0, 1)
    carrying = grasped * (2 + 2 * ((height - .015) / .1).clamp(0, 1)
                          + 2 * (1 - torch.tanh(goal_distance(env) / .1)))
    if env.cfg.task_profile_task == "lift":
        return approaching + closing + carrying
    command = env.command_manager.get_term("object_pose")
    placing = (command.stage == 2) * (1 - torch.tanh(goal_distance(env) / .03)) * (jaw / .8).clamp(0, 1)
    return approaching + closing * (command.stage < 2) + carrying + command.stage * 4 + placing * 2


def task_activity(env):
    maximum = 9. if env.cfg.task_profile_task == "lift" else 18.
    score = task_potential(env) / maximum
    if not torch.isfinite(score).all() or (score < 0).any() or (score > 1 + 1e-6).any():
        raise RuntimeError("任务进展数值超出配置范围")
    return score


class TaskProgressReward(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.previous[env_ids if env_ids is not None else slice(None)] = 0.

    def __call__(self, env):
        current = task_potential(env)
        current = torch.where(env.termination_manager.dones, 0., current)
        # 有限时间 episode 的终止势能为零，累计 shaping 保持任务的成功目标。
        shaping = 5 * (env.cfg.reward_discount * current - self.previous)
        self.previous.copy_(current)
        return shaping / env.step_dt


@configclass
class ProgressRewardsCfg:
    progress = RewardTermCfg(func=TaskProgressReward, weight=1.)
    task_activity = RewardTermCfg(func=task_activity, weight=.25)
    processed_action_change = RewardTermCfg(func=processed_action_change, weight=-.1)
    joint_vel = RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-4)
    success_bonus = RewardTermCfg(func=success_event, weight=1.)
