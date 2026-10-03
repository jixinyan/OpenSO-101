import argparse
import copy
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from openso101.rl.config import digest
from openso101.sim2sim.constrained_drive import ConstrainedImplicitDrive
from openso101.sim2sim.mujoco import JOINT_NAMES, JOINT_OFFSETS, build_model, jaw_forces
from openso101.sim2sim.recorded_physics import COMPONENT_FIELDS, RecordedPhysics


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--robot-model", type=Path, required=True)
parser.add_argument("--collision-bundle", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = args.source.resolve()
report = json.loads((source / "report.json").read_text())
states = json.loads((source / "initial_states.json").read_text())
for name, field in (("initial_states.json", "states_sha256"), ("trajectory.hdf5", "trace_sha256"),
                    ("initial_physics.hdf5", "initial_physics_sha256")):
    if digest(source / name) != report[field]:
        raise ValueError(f"原生来源文件 SHA256 不一致：{name}")
if report["task"] != "OpenSO101-Lift-v0" or report["task_profile"] != "grasp_v4":
    raise ValueError("实际抓取回放需要完整范围位置控制的原生 Lift")
with h5py.File(source / "initial_physics.hdf5") as stream:
    fields = {name: stream[name][:] for name in stream}
with h5py.File(source / "trajectory.hdf5") as stream:
    native = {name: stream[name][:] for name in (
        "joint_position", "joint_targets", "object_position_root", "jaw_forces", "active", "phase", "success")}
if any(not np.isfinite(value).all() for value in (*fields.values(), *native.values())):
    raise ValueError("原生来源包含无效数值")
metadata = dict(states["planner_physics"], physics_recording=states["physics_recording"], quaternion_order="wxyz")
template = build_model(args.robot_model, metadata, args.collision_bundle)
args.output.mkdir(parents=True, exist_ok=False)
parameters = states["task_parameters"]
substeps = round(states["control_dt"] / template.opt.timestep)
if substeps <= 0 or not np.isclose(substeps * template.opt.timestep, states["control_dt"]):
    raise ValueError("实际控制周期无法表示为完整 MuJoCo 物理步骤")
records = []
with h5py.File(args.output / "trajectory.hdf5", "x") as trajectory:
    for environment in range(report["episodes"]):
        model = copy.copy(template)
        data = mujoco.MjData(model)
        qpos_ids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
        dof_ids = [int(model.joint(name).dofadr[0]) for name in JOINT_NAMES]
        actuator_ids = [model.actuator(name).id for name in JOINT_NAMES]
        object_qpos = int(model.joint("object_free").qposadr[0])
        object_dof = int(model.joint("object_free").dofadr[0])
        object_id = model.body("object").id
        data.qpos[qpos_ids] = fields["joint_position"][0, environment] + JOINT_OFFSETS
        data.qvel[dof_ids] = fields["joint_velocity"][0, environment]
        data.qpos[object_qpos:object_qpos + 3] = fields["object_position_root"][0, environment]
        quaternion = fields["object_quaternion_root"][0, environment]
        data.qpos[object_qpos + 3:object_qpos + 7] = quaternion
        data.qvel[object_dof:object_dof + 3] = fields["object_linear_velocity_root"][0, environment]
        data.qvel[object_dof + 3:object_dof + 6] = Rotation.from_quat(quaternion, scalar_first=True).inv().apply(
            fields["object_angular_velocity_root"][0, environment])
        mujoco.mj_forward(model, data)
        physics = RecordedPhysics(model, metadata, fields, list(COMPONENT_FIELDS), environment)
        physics.apply(data, 0)
        drive = ConstrainedImplicitDrive(model, actuator_ids, qpos_ids, dof_ids)
        drive.configure(fields["joint_stiffness"][0, environment], fields["joint_damping"][0, environment],
                        fields["joint_vel_limits"][0, environment])
        selected = np.flatnonzero(native["active"][:, environment] & (native["phase"][:, environment] >= 0))
        if len(selected) < 2 or not np.array_equal(selected, np.arange(selected[0], selected[-1] + 1)):
            raise ValueError("原生首次抓取轨迹需要连续的完整控制步骤")
        if selected[0] != states["settling_steps"]:
            raise ValueError("原生实际初始物理状态与回放起点不一致")
        goal = np.asarray(states["environments"][environment]["goal_position_root"])
        hold = 0.
        success = False
        buffers = {name: [] for name in ("joint_position", "object_position_root", "jaw_forces", "success")}
        velocities, torques = [], []
        for step in selected:
            targets = native["joint_targets"][step, environment] + JOINT_OFFSETS
            for _ in range(substeps):
                drive.apply(data, targets)
                mujoco.mj_step(model, data)
                drive.verify(data)
                velocities.append(data.qvel[dof_ids].copy())
                torques.append(data.actuator_force[actuator_ids].copy())
            mujoco.mj_forward(model, data)
            position = data.xpos[object_id].copy()
            forces = jaw_forces(model, data)
            eligible = (position[2] > states["task_reference_height_root"] + parameters["minimal_height"]
                        and np.linalg.norm(position - goal) < parameters["goal_radius"]
                        and (forces > parameters["force_threshold"]).all())
            hold = hold + states["control_dt"] if eligible else 0.
            success |= hold >= parameters["settle_seconds"]
            for name, value in (("joint_position", data.qpos[qpos_ids] - JOINT_OFFSETS),
                                ("object_position_root", position), ("jaw_forces", forces), ("success", success)):
                buffers[name].append(np.asarray(value).copy())
        arrays = {name: np.asarray(value) for name, value in buffers.items()}
        group = trajectory.create_group(f"environment_{environment:06d}")
        for name, values in arrays.items():
            group.create_dataset(f"mujoco/{name}", data=values)
            group.create_dataset(f"isaac/{name}", data=native[name][selected, environment])
        group.create_dataset("joint_targets", data=native["joint_targets"][selected, environment])
        group.create_dataset("physics_steps/joint_velocity", data=np.asarray(velocities))
        group.create_dataset("physics_steps/actuator_force", data=np.asarray(torques))
        error = arrays["joint_position"] - native["joint_position"][selected, environment]
        records.append({"environment": environment, "native_success": report["environments"][environment]["success"],
                        "mujoco_success": bool(success), "control_steps": len(selected),
                        "maximum_object_height_root_m": float(arrays["object_position_root"][:, 2].max()),
                        "bilateral_contact_steps": int((arrays["jaw_forces"] > parameters["force_threshold"]).all(axis=-1).sum()),
                        "joint_position_rmse_rad": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
                        "maximum_object_position_error_m": float(np.linalg.norm(
                            arrays["object_position_root"] - native["object_position_root"][selected, environment], axis=-1).max()),
                        "recorded_physics": physics.report(), "constrained_drive": drive.report()})
        print(json.dumps(records[-1]), flush=True)
result = {"status": "native_scripted_action_replay_completed", "task": report["task"], "environments": records,
          "mujoco_successes": sum(item["mujoco_success"] for item in records), "episodes": len(records),
          "source_report_sha256": digest(source / "report.json"), "source_trace_sha256": report["trace_sha256"],
          "source_initial_physics_sha256": report["initial_physics_sha256"],
          "source_states_sha256": report["states_sha256"], "source_script_sha256": report["source_sha256"],
          "robot_model_sha256": digest(args.robot_model), "collision_bundle_sha256": digest(args.collision_bundle / "manifest.json"),
          "replay_source_sha256": digest(Path(__file__)), "mujoco_version": mujoco.__version__,
          "trajectory_sha256": digest(args.output / "trajectory.hdf5"), "control_dt": states["control_dt"],
          "physics_dt": template.opt.timestep, "action_source": "actual_native_joint_targets",
          "physics_equivalence_verified": False, "rl_policy_success_verified": False}
(args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False), flush=True)
