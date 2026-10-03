import torch
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass


class NormalizedJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._lower = limits[..., 0].clone()
        self._upper = limits[..., 1].clone()
        if self.cfg.clip is not None:
            self._lower = torch.maximum(self._lower, self._clip[..., 0])
            self._upper = torch.minimum(self._upper, self._clip[..., 1])
        if not torch.isfinite(limits).all() or (self._lower >= self._upper).any():
            raise ValueError("归一化关节控制需要有效的位置范围")
        self._scale = (self._upper - self._lower) / 2
        self._offset = (self._upper + self._lower) / 2

    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("关节位置动作需要有限数值")
        self._raw_actions[:] = actions
        self._processed_actions = self._offset + actions.clamp(-1, 1) * self._scale

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._processed_actions[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self._raw_actions[ids] = ((self._processed_actions[ids] - self._offset[ids]) / self._scale[ids]).clamp(-1, 1)


@configclass
class NormalizedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = NormalizedJointPositionAction
    use_default_offset: bool = False
