import numpy as np


class VelocityLimitedServo:
    def __init__(self, model, actuator_ids, qpos_ids):
        self.model = model
        self.actuator_ids = np.asarray(actuator_ids)
        self.qpos_ids = np.asarray(qpos_ids)
        self.stiffness = None
        self.damping = None
        self.limits = None

    def configure(self, stiffness, damping, limits):
        values = [np.asarray(value, dtype=np.float64) for value in (stiffness, damping, limits)]
        if any(value.shape != self.actuator_ids.shape or not np.isfinite(value).all() for value in values):
            raise ValueError("velocity-servo 参数需要与关节数量一致且有限")
        self.stiffness, self.damping, self.limits = values
        if (self.stiffness < 0).any() or (self.damping <= 0).any() or (self.limits <= 0).any():
            raise ValueError("velocity-servo 要求非负 stiffness、正数 damping 与速度限制")
        ids = self.actuator_ids
        self.model.actuator_gainprm[ids, :] = 0
        self.model.actuator_gainprm[ids, 0] = self.damping
        self.model.actuator_biasprm[ids, :] = 0
        self.model.actuator_biasprm[ids, 2] = -self.damping
        self.model.actuator_ctrlrange[ids, 0] = -self.limits
        self.model.actuator_ctrlrange[ids, 1] = self.limits
        self.model.actuator_ctrllimited[ids] = True

    def apply(self, data, targets):
        targets = np.asarray(targets, dtype=np.float64)
        if targets.shape != self.qpos_ids.shape or not np.isfinite(targets).all():
            raise ValueError("velocity-servo 关节目标需要与关节数量一致且有限")
        if self.limits is None:
            raise RuntimeError("velocity-servo 需要配置实际参数")
        # 位置反馈生成受限速度目标，原生 actuator 根据速度误差施加力矩。
        desired = self.stiffness / self.damping * (targets - data.qpos[self.qpos_ids])
        data.ctrl[self.actuator_ids] = np.clip(desired, -self.limits, self.limits)
        return data.ctrl[self.actuator_ids].copy()
