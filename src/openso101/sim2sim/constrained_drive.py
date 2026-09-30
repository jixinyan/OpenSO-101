import mujoco
import numpy as np
import osqp
from scipy import sparse


class ConstrainedImplicitDrive:
    def __init__(self, model, actuator_ids, qpos_ids, dof_ids):
        self.model = model
        self.actuator_ids = np.asarray(actuator_ids)
        self.qpos_ids = np.asarray(qpos_ids)
        self.dof_ids = np.asarray(dof_ids)
        self.trial = mujoco.MjData(model)
        self.mass = np.empty((model.nv, model.nv))
        self.limits = None
        self.steps = 0
        self.max_prediction_error = 0.
        self.max_actual_speed = np.zeros(len(actuator_ids))
        self.max_actual_force = np.zeros(len(actuator_ids))
        self.max_iterations = 0
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        if np.any(model.dof_damping[self.dof_ids] != 0):
            raise ValueError("constrained drive 需要零被动关节 damping")
        if (np.any(model.actuator_trntype[self.actuator_ids] != mujoco.mjtTrn.mjTRN_JOINT)
                or not np.array_equal(model.actuator_gear[self.actuator_ids], np.tile([1., 0, 0, 0, 0, 0], (len(actuator_ids), 1)))):
            raise ValueError("constrained drive 需要单位 gear 的直接 joint actuator")
        model.actuator_gainprm[self.actuator_ids] = 0
        model.actuator_gainprm[self.actuator_ids, 0] = 1
        model.actuator_biasprm[self.actuator_ids] = 0
        model.actuator_ctrllimited[self.actuator_ids] = False
        if not model.actuator_forcelimited[self.actuator_ids].all():
            raise ValueError("constrained drive 需要实际 actuator force limits")

    def configure(self, stiffness, damping, limits):
        values = [np.asarray(value, dtype=np.float64) for value in (stiffness, damping, limits)]
        if any(value.shape != self.actuator_ids.shape or not np.isfinite(value).all() for value in values):
            raise ValueError("constrained drive 参数形状或数值无效")
        self.stiffness, self.damping, self.limits = values
        if (self.stiffness < 0).any() or (self.damping <= 0).any() or (self.limits <= 1e-6).any():
            raise ValueError("constrained drive 需要非负 stiffness、正数 damping 与速度限制")

    def _predict(self, data, torque):
        # 预测使用独立的真实 MuJoCo 步骤，运行状态保持原值。
        mujoco.mj_copyData(self.trial, self.model, data)
        self.trial.ctrl[self.actuator_ids] = torque
        mujoco.mj_step(self.model, self.trial)
        if not np.isfinite(self.trial.qpos).all() or not np.isfinite(self.trial.qvel).all() or any(warning.number for warning in self.trial.warning):
            raise RuntimeError("constrained drive 的物理预测产生无效状态或警告")
        return self.trial.qvel[self.dof_ids].copy()

    def apply(self, data, targets):
        targets = np.asarray(targets, dtype=np.float64)
        if self.limits is None or targets.shape != self.qpos_ids.shape or not np.isfinite(targets).all():
            raise ValueError("constrained drive 需要完整配置及有限关节目标")
        mujoco.mj_forward(self.model, data)
        mujoco.mj_fullM(self.model, data, self.mass)
        inertia = self.mass[np.ix_(self.dof_ids, self.dof_ids)]
        timestep = self.model.opt.timestep
        metric = inertia + np.diag(timestep * self.damping + timestep**2 * self.stiffness)
        torque_bounds = self.model.actuator_forcerange[self.actuator_ids]
        limits = self.limits - 1e-6
        center = np.zeros(len(self.actuator_ids))
        radius = np.max(np.abs(torque_bounds), axis=1)
        for iteration in range(32):
            base = self._predict(data, center)
            jacobian = np.empty((len(center), len(center)))
            for index in range(len(center)):
                perturbation = center.copy()
                delta = -1e-4 if center[index] + 1e-4 > torque_bounds[index, 1] else 1e-4
                perturbation[index] += delta
                jacobian[:, index] = (self._predict(data, perturbation) - base) / delta
            intercept = base - jacobian @ center
            error_force = self.stiffness * (targets - data.qpos[self.qpos_ids])
            feedback = self.damping + timestep * self.stiffness
            # 完整力矩反馈保留夹爪压紧力，实际速度约束通过物理响应矩阵求解。
            response = np.eye(len(center)) + feedback[:, None] * jacobian
            residual = feedback * intercept - error_force
            quadratic = response.T @ np.linalg.solve(metric, response)
            linear = response.T @ np.linalg.solve(metric, residual)
            scale = np.max(np.abs(quadratic))
            if not np.isfinite(scale) or scale <= 0:
                raise RuntimeError("constrained drive 的控制响应矩阵无效")
            solver = osqp.OSQP()
            solver.setup(P=sparse.csc_matrix(quadratic / scale), q=linear / scale,
                         A=sparse.csc_matrix(np.vstack([jacobian, np.eye(len(center))])),
                         l=np.concatenate([-limits - intercept, np.maximum(torque_bounds[:, 0], center - radius)]),
                         u=np.concatenate([limits - intercept, np.minimum(torque_bounds[:, 1], center + radius)]),
                         verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=10000,
                         adaptive_rho_interval=25)
            solution = solver.solve(raise_error=True)
            torque = solution.x
            if not np.isfinite(torque).all() or np.any(torque < torque_bounds[:, 0] - 1e-8) or np.any(torque > torque_bounds[:, 1] + 1e-8):
                raise RuntimeError("constrained drive 的力矩解无效")
            predicted = self._predict(data, torque)
            if np.all(np.abs(predicted) <= limits + 1e-8):
                self.predicted_velocity = predicted
                self.max_iterations = max(self.max_iterations, iteration + 1)
                data.ctrl[self.actuator_ids] = torque
                return predicted.copy()
            center = torque
            radius *= .5
        raise RuntimeError(f"constrained drive 未找到满足实际速度限制的力矩: time={data.time}, velocity={predicted}")

    def verify(self, data):
        velocity = data.qvel[self.dof_ids]
        force = data.actuator_force[self.actuator_ids]
        if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                or any(warning.number for warning in data.warning)):
            raise RuntimeError("constrained drive 产生无效运行状态或警告")
        if np.any(np.abs(velocity) > self.limits + 1e-8):
            raise RuntimeError("constrained drive 的实际关节速度超过限制")
        bounds = self.model.actuator_forcerange[self.actuator_ids]
        if np.any(force < bounds[:, 0] - 1e-8) or np.any(force > bounds[:, 1] + 1e-8):
            raise RuntimeError("constrained drive 的实际力矩超过限制")
        error = float(np.max(np.abs(velocity - self.predicted_velocity)))
        if error > 1e-8:
            raise RuntimeError("constrained drive 的预测与实际速度不一致")
        self.max_prediction_error = max(self.max_prediction_error, error)
        self.max_actual_speed = np.maximum(self.max_actual_speed, np.abs(velocity))
        self.max_actual_force = np.maximum(self.max_actual_force, np.abs(force))
        self.steps += 1

    def report(self):
        return {"physics_steps": self.steps, "max_actual_speed_rad_s": self.max_actual_speed.tolist(),
                "max_actuator_force_nm": self.max_actual_force.tolist(),
                "maximum_prediction_error_rad_s": self.max_prediction_error,
                "maximum_optimization_iterations": self.max_iterations,
                "actual_velocity_limits_verified": self.steps > 0,
                "actual_torque_limits_verified": self.steps > 0,
                "controller": "OSQP_implicit_PD_with_actual_MuJoCo_step_constraints",
                "integrator": "semi_implicit_Euler", "state_velocity_modified": False}
