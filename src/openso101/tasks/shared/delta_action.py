import torch
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass


class JointDeltaAction(JointPositionAction):
    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("关节增量动作需要有限数值")
        self._raw_actions[:] = actions
        # 每个控制步骤读取一次关节位置，所有物理步骤使用同一个目标。
        self._offset = self._asset.data.joint_pos[:, self._joint_ids].clone()
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        lower, upper = limits[..., 0], limits[..., 1]
        if self.cfg.clip is not None:
            lower = torch.maximum(lower, self._clip[..., 0])
            upper = torch.minimum(upper, self._clip[..., 1])
        self._processed_actions = (self._offset + actions.clamp(-1, 1) * self._scale).clamp(lower, upper)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.
        self._processed_actions[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]


@configclass
class JointDeltaActionCfg(JointPositionActionCfg):
    class_type: type = JointDeltaAction
    use_default_offset: bool = False
