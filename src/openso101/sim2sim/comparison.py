import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from openso101.rl.config import digest
from openso101.rl.portable import PortablePolicy
from .mujoco import JOINT_NAMES, JOINT_OFFSETS, build_model, check_kinematics, jaw_forces


def compare(args):
    if args.steps <= 1 or args.episodes <= 0:
        raise ValueError("steps 必须大于 1，episodes 必须为正数")
    policy_folder = Path(args.policy).resolve()
    robot_model = Path(args.robot_model).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    policy = PortablePolicy(policy_folder)
    metadata = policy.metadata
    reference_path = policy_folder / "isaac_validation.hdf5"
    with h5py.File(reference_path, "r") as trace:
        positions = trace["joint_position"][:]
        if positions.ndim != 3 or positions.shape[2] != 6:
            raise ValueError("Isaac 关节位置需要具有 steps×environments×6 形状")
        if args.steps > len(positions) or args.episodes > positions.shape[1]:
            raise ValueError("请求数量超过实际 Isaac 记录数量")
        kinematics = check_kinematics(mujoco.MjModel.from_xml_path(str(robot_model)), trace)
        fields = {name: trace[name][:args.steps, :args.episodes] for name in (
            "joint_position", "joint_velocity", "joint_targets", "object_position_root",
            "object_quaternion_root", "jaw_forces", "terminated", "truncated",
        )}
        physics_fields = ("joint_stiffness", "joint_damping", "joint_armature", "joint_friction_coeff", "joint_vel_limits")
        if args.recorded_pd and not all(name in trace for name in physics_fields):
            raise ValueError("recorded-pd 需要实际 Isaac 物理参数记录")
        for name in physics_fields + ("object_linear_velocity_root", "object_angular_velocity_root"):
            if name in trace:
                fields[name] = trace[name][:args.steps, :args.episodes]
    for name, values in fields.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Isaac 原始数据含有无效数值: {name}")
    for name in ("joint_position", "joint_velocity", "joint_targets"):
        if fields[name].shape != (args.steps, args.episodes, 6):
            raise ValueError(f"Isaac 字段形状不一致: {name}")
    for name in physics_fields:
        if name in fields and fields[name].shape != (args.steps, args.episodes, 6):
            raise ValueError(f"Isaac 物理参数形状不一致: {name}")
    velocity_available = all(name in fields for name in ("object_linear_velocity_root", "object_angular_velocity_root"))
    if any(name in fields for name in ("object_linear_velocity_root", "object_angular_velocity_root")) and not velocity_available:
        raise ValueError("物体线速度与角速度需要同时记录")
    if velocity_available and any(fields[name].shape != (args.steps, args.episodes, 3) for name in ("object_linear_velocity_root", "object_angular_velocity_root")):
        raise ValueError("物体速度记录形状不一致")
    for name, width in (("object_position_root", 3), ("object_quaternion_root", 4), ("jaw_forces", 2)):
        if fields[name].shape != (args.steps, args.episodes, width):
            raise ValueError(f"Isaac 字段形状不一致: {name}")
    for name in ("terminated", "truncated"):
        if fields[name].shape != (args.steps, args.episodes):
            raise ValueError(f"Isaac 终止标记形状不一致: {name}")
    model = build_model(robot_model, metadata)
    data = mujoco.MjData(model)
    joint_qpos_ids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    joint_dof_ids = [int(model.joint(name).dofadr[0]) for name in JOINT_NAMES]
    actuator_ids = [model.actuator(name).id for name in JOINT_NAMES]
    object_qpos_id = int(model.joint("object_free").qposadr[0])
    object_dof_id = int(model.joint("object_free").dofadr[0])
    object_id = model.body("object").id
    control_dt = metadata["control_dt"]
    substeps = round(control_dt / model.opt.timestep)
    if substeps <= 0 or not np.isclose(substeps * model.opt.timestep, control_dt):
        raise ValueError("MuJoCo physics_dt 无法表示实际 control_dt")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with h5py.File(output / "comparison.hdf5", "w") as trajectory:
        for episode in range(args.episodes):
            # 只比较首次 episode 的连续片段，停止于源记录的终止步骤。
            boundaries = np.flatnonzero(fields["terminated"][:, episode] | fields["truncated"][:, episode])
            steps = min(args.steps, int(boundaries[0]) + 1) if len(boundaries) else args.steps
            if steps <= 1:
                raise ValueError(f"源环境 {episode} 的连续记录不足两个步骤")
            mujoco.mj_resetData(model, data)
            data.qpos[joint_qpos_ids] = fields["joint_position"][0, episode] + JOINT_OFFSETS
            data.qvel[joint_dof_ids] = fields["joint_velocity"][0, episode]
            data.qpos[object_qpos_id:object_qpos_id + 3] = fields["object_position_root"][0, episode]
            data.qpos[object_qpos_id + 3:object_qpos_id + 7] = fields["object_quaternion_root"][0, episode]
            if velocity_available:
                data.qvel[object_dof_id:object_dof_id + 3] = fields["object_linear_velocity_root"][0, episode]
                # MuJoCo freejoint 的角速度使用 body 坐标。
                rotation = np.zeros(9)
                mujoco.mju_quat2Mat(rotation, data.qpos[object_qpos_id + 3:object_qpos_id + 7])
                data.qvel[object_dof_id + 3:object_dof_id + 6] = rotation.reshape(3, 3).T @ fields["object_angular_velocity_root"][0, episode]
            mujoco.mj_forward(model, data)
            initial_position_error = float(np.max(np.abs(data.qpos[joint_qpos_ids] - JOINT_OFFSETS - fields["joint_position"][0, episode])))
            initial_velocity_error = float(np.max(np.abs(data.qvel[joint_dof_ids] - fields["joint_velocity"][0, episode])))
            if initial_position_error > 1e-6 or initial_velocity_error > 1e-6:
                raise ValueError("MuJoCo 初始关节状态与源数据不一致")
            buffers = {name: [] for name in ("joint_position", "joint_velocity", "object_position_root", "jaw_forces")}
            for step in range(steps):
                buffers["joint_position"].append(data.qpos[joint_qpos_ids].copy() - JOINT_OFFSETS)
                buffers["joint_velocity"].append(data.qvel[joint_dof_ids].copy())
                buffers["object_position_root"].append(data.xpos[object_id].copy())
                buffers["jaw_forces"].append(jaw_forces(model, data))
                data.ctrl[actuator_ids] = fields["joint_targets"][step, episode] + JOINT_OFFSETS
                if args.recorded_pd:
                    stiffness = fields["joint_stiffness"][step, episode]
                    damping = fields["joint_damping"][step, episode]
                    if (stiffness < 0).any() or (damping < 0).any():
                        raise ValueError("实际 PD 参数需要非负")
                    model.actuator_gainprm[actuator_ids, 0] = stiffness
                    model.actuator_biasprm[actuator_ids, 1] = -stiffness
                    model.actuator_biasprm[actuator_ids, 2] = -damping
                for _ in range(substeps):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(warning.number for warning in data.warning):
                    raise RuntimeError("MuJoCo 动力学比较产生无效状态或警告")
            arrays = {name: np.asarray(values) for name, values in buffers.items()}
            position_error = arrays["joint_position"] - fields["joint_position"][:steps, episode]
            velocity_error = arrays["joint_velocity"] - fields["joint_velocity"][:steps, episode]
            object_error = np.linalg.norm(arrays["object_position_root"] - fields["object_position_root"][:steps, episode], axis=-1)
            group = trajectory.create_group(f"environment_{episode:06d}")
            for name, values in arrays.items():
                group.create_dataset(f"mujoco/{name}", data=values)
                group.create_dataset(f"isaac/{name}", data=fields[name][:steps, episode])
            group.create_dataset("joint_targets", data=fields["joint_targets"][:steps, episode])
            for name in physics_fields:
                if name in fields:
                    group.create_dataset(f"isaac/{name}", data=fields[name][:steps, episode])
            peak_step, peak_joint = np.unravel_index(np.argmax(np.abs(position_error)), position_error.shape)
            records.append({
                "source_env_index": episode, "steps": steps,
                "compared_state_samples": steps, "duration_seconds": steps * control_dt,
                "source_episode_boundary": bool(len(boundaries)),
                "initial_joint_position_error_rad": initial_position_error,
                "initial_joint_velocity_error_rad_s": initial_velocity_error,
                "joint_position_rmse_rad": np.sqrt(np.mean(position_error ** 2, axis=0)).tolist(),
                "joint_position_max_error_rad": np.max(np.abs(position_error), axis=0).tolist(),
                "joint_velocity_rmse_rad_s": np.sqrt(np.mean(velocity_error ** 2, axis=0)).tolist(),
                "object_position_mean_error_m": float(np.mean(object_error)),
                "object_position_max_error_m": float(np.max(object_error)),
                "isaac_max_jaw_force_n": np.max(fields["jaw_forces"][:steps, episode], axis=0).tolist(),
                "mujoco_max_jaw_force_n": np.max(arrays["jaw_forces"], axis=0).tolist(),
                "peak_position_error": {
                    "step": int(peak_step), "time_seconds": float(peak_step * control_dt),
                    "joint": JOINT_NAMES[peak_joint], "error_rad": float(position_error[peak_step, peak_joint]),
                    "isaac_position_rad": float(fields["joint_position"][peak_step, episode, peak_joint]),
                    "mujoco_position_rad": float(arrays["joint_position"][peak_step, peak_joint]),
                },
                "isaac_max_joint_speed_rad_s": np.max(np.abs(fields["joint_velocity"][:steps, episode]), axis=0).tolist(),
                "mujoco_max_joint_speed_rad_s": np.max(np.abs(arrays["joint_velocity"]), axis=0).tolist(),
                "recorded_physics_initial": {name: fields[name][0, episode].tolist() for name in physics_fields if name in fields},
            })
    report = {
        "status": "mujoco_recorded_action_comparison_completed",
        "task": metadata["task_id"], "task_profile": metadata.get("task_profile", "default"),
        "joint_names": list(JOINT_NAMES), "environments": records,
        "control_dt": control_dt, "physics_dt": model.opt.timestep,
        "kinematics": kinematics, "mujoco_version": mujoco.__version__,
        "policy_sha256": metadata["files"]["policy.pt"], "policy_metadata_sha256": digest(policy_folder / "policy.json"),
        "robot_model_sha256": digest(robot_model),
        "robot_meshes": {path.name: digest(path) for path in sorted((robot_model.parent / "assets").glob("*.stl"))},
        "isaac_trace_sha256": digest(reference_path), "trajectory_sha256": digest(output / "comparison.hdf5"),
        "comparison_source_sha256": digest(Path(__file__)),
        "model_builder_sha256": digest(Path(__file__).with_name("mujoco.py")),
        "action_source": "actual_Isaac_ActionManager_joint_targets",
        "sampling": "state_before_each_control_step",
        "pd_source": "recorded_Isaac_per_environment" if args.recorded_pd else "nominal_policy_metadata",
        "source_object_velocity_available": velocity_available,
        "mujoco_initial_object_velocity": "recorded_Isaac" if velocity_available else "zero",
        "physics_equivalence_verified": False, "task_success_verified": False,
        "limitations": ["Isaac 实际质量、惯性与接触材质未记录", "两个模拟器的接触、惯性、摩擦和速度限制需要独立测量"]
        + ([] if velocity_available else ["Isaac 物体初始速度未记录"]),
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))
    return 0
