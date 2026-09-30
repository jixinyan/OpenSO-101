import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .mujoco import JOINT_NAMES, JOINT_OFFSETS


COMPONENT_FIELDS = {
    "bodies": ("robot_body_mass", "robot_body_inertia", "robot_body_com", "robot_body_position_root",
               "robot_body_quaternion_root", "object_body_mass", "object_body_inertia", "object_body_com"),
    "gravity": ("scene_gravity",),
    "armature": ("joint_armature",),
    "friction": ("joint_friction_coeff",),
}


class RecordedPhysics:
    def __init__(self, model, metadata, fields, components, environment):
        self.model = model
        self.fields = fields
        self.components = components
        self.environment = environment
        self.reference = mujoco.MjData(model)
        self.constants = mujoco.MjData(model)
        self.qpos_ids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
        self.dof_ids = [int(model.joint(name).dofadr[0]) for name in JOINT_NAMES]
        self.object_id = model.body("object").id
        self.object_qpos_id = int(model.joint("object_free").qposadr[0])
        self.max_com_error = 0.0
        self.max_inertia_error = 0.0
        self.max_mass_error = 0.0
        self.verified_steps = 0
        if "bodies" in components:
            recording = metadata["physics_recording"]
            if (recording["inertia_frame"] != "body_prim_at_center_of_mass"
                    or recording["inertia_matrix_order"] != "column_major"
                    or recording["com_pose_frame"] != "body_prim"
                    or recording["com_quaternion_order"] != "xyzw"
                    or metadata["quaternion_order"] != "wxyz"):
                raise ValueError("实际实体参数需要已定义的 body COM 惯性坐标")
            self.body_names = recording["robot_body_names"]
            self.body_ids = [model.body("moving_jaw_so101_v1" if name == "jaw" else name).id for name in self.body_names]
            self._reference_pose(0)
            native_rotation = Rotation.from_quat(fields["robot_body_quaternion_root"][0, environment], scalar_first=True).as_matrix()
            model_rotation = self.reference.xmat[self.body_ids].reshape(-1, 3, 3)
            self.frame_rotation = model_rotation.swapaxes(-1, -2) @ native_rotation
            delta = fields["robot_body_position_root"][0, environment] - self.reference.xpos[self.body_ids]
            self.frame_translation = np.einsum("bij,bj->bi", model_rotation.swapaxes(-1, -2), delta)

    def _reference_pose(self, step):
        self.reference.qpos[self.qpos_ids] = self.fields["joint_position"][step, self.environment] + JOINT_OFFSETS
        address = self.object_qpos_id
        self.reference.qpos[address:address + 3] = self.fields["object_position_root"][step, self.environment]
        self.reference.qpos[address + 3:address + 7] = self.fields["object_quaternion_root"][step, self.environment]
        mujoco.mj_forward(self.model, self.reference)

    def _body_parameters(self, mass, inertia, com, body_ids):
        inertia = np.asarray(inertia, dtype=np.float64)
        if not np.allclose(inertia, inertia.swapaxes(-1, -2), atol=1e-10, rtol=1e-5):
            raise ValueError("实际惯性矩阵需要对称")
        inertia = (inertia + inertia.swapaxes(-1, -2)) / 2
        moments, axes = np.linalg.eigh(inertia)
        if (np.asarray(mass) <= 0).any() or (moments <= 0).any():
            raise ValueError("实际质量与主惯性矩需要正数")
        if (moments[:, 0] + moments[:, 1] < moments[:, 2] - 1e-10).any():
            raise ValueError("实际惯性矩需要满足刚体主惯性矩条件")
        # 正交特征向量组成右手坐标系，供 MuJoCo 的惯性 quaternion 使用。
        axes[:, :, -1] *= np.where(np.linalg.det(axes) < 0, -1, 1)[:, None]
        quaternion = Rotation.from_matrix(axes).as_quat(scalar_first=True)
        self.model.body_mass[body_ids] = np.asarray(mass).reshape(-1)
        self.model.body_ipos[body_ids] = com
        self.model.body_inertia[body_ids] = moments
        self.model.body_iquat[body_ids] = quaternion
        rotations = Rotation.from_quat(self.model.body_iquat[body_ids], scalar_first=True).as_matrix()
        actual = (rotations * self.model.body_inertia[body_ids, None, :]) @ rotations.swapaxes(-1, -2)
        error = np.max(np.linalg.norm(actual - inertia, axis=(-1, -2)) / np.linalg.norm(inertia, axis=(-1, -2)))
        if error > 1e-10 or not np.array_equal(self.model.body_mass[body_ids], np.asarray(mass).reshape(-1)):
            raise RuntimeError("MuJoCo 实际质量与惯性写入检查失败")

    def apply(self, data, step):
        environment = self.environment
        qpos, qvel, time = data.qpos.copy(), data.qvel.copy(), data.time
        if "bodies" in self.components:
            fields = self.fields
            inertia = fields["robot_body_inertia"][step, environment].reshape(-1, 3, 3).swapaxes(-1, -2)
            converted = self.frame_rotation @ inertia @ self.frame_rotation.swapaxes(-1, -2)
            com = self.frame_translation + np.einsum("bij,bj->bi", self.frame_rotation, fields["robot_body_com"][step, environment, :, :3])
            self._body_parameters(fields["robot_body_mass"][step, environment], converted, com, self.body_ids)
            self._body_parameters(
                fields["object_body_mass"][step, environment],
                fields["object_body_inertia"][step, environment].reshape(1, 3, 3).swapaxes(-1, -2),
                fields["object_body_com"][step, environment, :3][None], [self.object_id],
            )
        if "gravity" in self.components:
            self.model.opt.gravity[:] = self.fields["scene_gravity"][step]
        if "armature" in self.components:
            armature = self.fields["joint_armature"][step, environment]
            if (armature < 0).any():
                raise ValueError("实际 armature 必须非负")
            self.model.dof_armature[self.dof_ids] = armature
        if "friction" in self.components:
            if np.any(self.fields["joint_friction_coeff"][step, environment] != 0):
                raise ValueError("实际非零摩擦系数需要独立的 frictionloss 转换验证")
            self.model.dof_frictionloss[self.dof_ids] = 0
        # 使用独立 MjData 重算 qpos0 处的模型常数，运行状态保持原值。
        mujoco.mj_setConst(self.model, self.constants)
        mujoco.mj_forward(self.model, data)
        if not np.array_equal(qpos, data.qpos) or not np.array_equal(qvel, data.qvel) or time != data.time:
            raise RuntimeError("物理参数更新改变了运行状态")
        if "gravity" in self.components and not np.array_equal(self.model.opt.gravity, self.fields["scene_gravity"][step]):
            raise RuntimeError("实际重力写入检查失败")
        if "armature" in self.components and not np.array_equal(self.model.dof_armature[self.dof_ids], self.fields["joint_armature"][step, environment]):
            raise RuntimeError("实际 armature 写入检查失败")
        if "friction" in self.components and np.any(self.model.dof_frictionloss[self.dof_ids] != 0):
            raise RuntimeError("实际零摩擦写入检查失败")
        if "bodies" in self.components:
            self._reference_pose(step)
            native_rotation = Rotation.from_quat(self.fields["robot_body_quaternion_root"][step, environment], scalar_first=True).as_matrix()
            native_com = self.fields["robot_body_position_root"][step, environment] + np.einsum("bij,bj->bi", native_rotation, self.fields["robot_body_com"][step, environment, :, :3])
            com_error = float(np.max(np.linalg.norm(native_com - self.reference.xipos[self.body_ids], axis=-1)))
            native_inertia = native_rotation @ inertia @ native_rotation.swapaxes(-1, -2)
            actual_rotation = self.reference.ximat[self.body_ids].reshape(-1, 3, 3)
            actual_inertia = (actual_rotation * self.model.body_inertia[self.body_ids, None, :]) @ actual_rotation.swapaxes(-1, -2)
            inertia_error = float(np.max(np.linalg.norm(actual_inertia - native_inertia, axis=(-1, -2)) / np.linalg.norm(native_inertia, axis=(-1, -2))))
            if com_error > 1e-5 or inertia_error > 1e-4:
                raise RuntimeError("共同坐标中的实际 COM 或惯性未通过检查")
            self.max_com_error = max(self.max_com_error, com_error)
            self.max_inertia_error = max(self.max_inertia_error, inertia_error)
            self.max_mass_error = max(self.max_mass_error, float(np.max(np.abs(self.model.body_mass[self.body_ids] - self.fields["robot_body_mass"][step, environment]))))
        self.verified_steps += 1

    def snapshot(self):
        return {"body_mass": self.model.body_mass.copy(), "body_ipos": self.model.body_ipos.copy(),
                "body_inertia": self.model.body_inertia.copy(), "body_iquat": self.model.body_iquat.copy(),
                "joint_armature": self.model.dof_armature[self.dof_ids].copy(),
                "joint_frictionloss": self.model.dof_frictionloss[self.dof_ids].copy(), "gravity": self.model.opt.gravity.copy()}

    def report(self):
        return {"components": self.components, "verified_control_steps": self.verified_steps,
                "maximum_source_com_error_m": self.max_com_error,
                "maximum_source_inertia_relative_error": self.max_inertia_error,
                "maximum_mass_error_kg": self.max_mass_error, "running_state_preserved_verified": True,
                "parameter_readback_verified": True}
