import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from openso101.rl.config import digest
from openso101.sim2sim.mujoco import JOINT_NAMES, JOINT_OFFSETS, build_model
from openso101.sim2sim.recorded_physics import COMPONENT_FIELDS, RecordedPhysics
from openso101.sim2sim.velocity_servo import VelocityLimitedServo


def experiment(robot_path, metadata, fields, timestep, velocity_servo, contacts, effort, table_geometry):
    if table_geometry is not None:
        metadata = metadata | {"table_geometry": table_geometry, "table_height_root": table_geometry["top_height_root"]}
    model = build_model(robot_path, metadata)
    ids = [model.joint(name).id for name in JOINT_NAMES]
    qids = [int(model.jnt_qposadr[index]) for index in ids]
    dids = [int(model.jnt_dofadr[index]) for index in ids]
    aids = [model.actuator(name).id for name in JOINT_NAMES]
    substeps = round(metadata["control_dt"] / timestep)
    if not np.isclose(substeps * timestep, metadata["control_dt"]):
        raise ValueError("时间步长需要完整表示源控制周期")
    object_id = model.body("object").id
    oq = int(model.joint("object_free").qposadr[0])
    ov = int(model.joint("object_free").dofadr[0])
    records = []
    arrays = {}
    for environment in range(fields["joint_position"].shape[1]):
        model = build_model(robot_path, metadata)
        model.opt.timestep = timestep
        model.actuator_forcerange[aids] = [-effort, effort]
        if not contacts:
            robot_geoms = np.flatnonzero(model.geom_contype == 1)
            model.geom_contype[robot_geoms] = 0
            model.geom_conaffinity[robot_geoms] = 0
        data = mujoco.MjData(model)
        data.qpos[qids] = fields["joint_position"][0, environment] + JOINT_OFFSETS
        data.qvel[dids] = fields["joint_velocity"][0, environment]
        data.qpos[oq:oq+3] = fields["object_position_root"][0, environment]
        data.qpos[oq+3:oq+7] = fields["object_quaternion_root"][0, environment]
        data.qvel[ov:ov+3] = fields["object_linear_velocity_root"][0, environment]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[oq+3:oq+7])
        data.qvel[ov+3:ov+6] = rotation.reshape(3, 3).T @ fields["object_angular_velocity_root"][0, environment]
        mujoco.mj_forward(model, data)
        physics = RecordedPhysics(model, metadata, fields, list(COMPONENT_FIELDS), environment)
        servo = VelocityLimitedServo(model, aids, qids) if velocity_servo else None
        boundary = np.flatnonzero(fields["terminated"][:, environment] | fields["truncated"][:, environment])
        steps = int(boundary[0])+1 if len(boundary) else len(fields["joint_position"])
        positions, objects, velocities, torques = [], [], [], []
        contact_count = 0
        for step in range(steps):
            physics.apply(data, step)
            positions.append(data.qpos[qids].copy() - JOINT_OFFSETS)
            objects.append(data.xpos[object_id].copy())
            stiffness = fields["joint_stiffness"][step, environment]
            damping = fields["joint_damping"][step, environment]
            target = fields["joint_targets"][step, environment] + JOINT_OFFSETS
            if servo:
                servo.configure(stiffness, damping, fields["joint_vel_limits"][step, environment])
            else:
                model.actuator_gainprm[aids, 0] = stiffness
                model.actuator_biasprm[aids, 1] = -stiffness
                model.actuator_biasprm[aids, 2] = -damping
                data.ctrl[aids] = target
            for _ in range(substeps):
                if servo:
                    servo.apply(data, target)
                mujoco.mj_step(model, data)
                velocities.append(data.qvel[dids].copy())
                torques.append(data.actuator_force[aids].copy())
                contact_count += sum(model.geom_contype[data.contact[index].geom1] == 1 or model.geom_contype[data.contact[index].geom2] == 1 for index in range(data.ncon))
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(warning.number for warning in data.warning):
                    raise RuntimeError("实际动作实验产生无效状态或警告")
            mujoco.mj_forward(model, data)
        positions, objects, velocities, torques = map(np.asarray, (positions, objects, velocities, torques))
        error = positions - fields["joint_position"][:steps, environment]
        object_error = np.linalg.norm(objects - fields["object_position_root"][:steps, environment], axis=-1)
        if np.max(np.abs(torques)) > effort + 1e-8:
            raise RuntimeError("原生力矩超过设置上限")
        peak = np.unravel_index(np.argmax(np.abs(velocities)), velocities.shape)
        record = {"environment": environment, "control_steps": steps, "physics_steps": steps*substeps,
                  "joint_position_rmse_rad": float(np.sqrt(np.mean(error**2))),
                  "maximum_joint_error_rad": float(np.abs(error).max()),
                  "maximum_speed_rad_s": float(np.abs(velocities).max()),
                  "maximum_actuator_torque_nm": float(np.abs(torques).max()),
                  "torque_limit_fraction": float(np.isclose(np.abs(torques), effort, atol=1e-7).mean()),
                  "maximum_object_error_m": float(object_error.max()),
                  "robot_contact_samples": int(contact_count),
                  "peak_speed_joint": JOINT_NAMES[peak[1]], "peak_speed_time_s": (peak[0]+1)*timestep,
                  "parameters": physics.report()}
        records.append(record)
        for name, array in (("joint_position", positions), ("object_position", objects), ("physics_velocity", velocities), ("physics_torque", torques)):
            arrays[f"env_{environment}_{name}"] = array
    return records, arrays


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--geometry-input", type=Path)
parser.add_argument("--measured-table-only", action="store_true")
args = parser.parse_args()
if args.output.exists() or (args.measured_table_only and args.geometry_input is None):
    raise ValueError("输出目录需要尚未存在，桌面专项检查需要原生场景报告")
root = Path(__file__).resolve().parents[1]
args.output.mkdir(parents=True, exist_ok=False)
conditions = [(f"velocity_dt_{dt:g}", dt, True, True, 30., None) for dt in (.01, .002, .001, .0005)]
conditions += [("position_dt_0.002", .002, False, True, 30., None),
               ("position_dt_0.0005", .0005, False, True, 30., None),
               ("velocity_no_contact", .002, True, False, 30., None),
               ("velocity_effort_3.35", .002, True, True, 3.35, None)]
if args.geometry_input:
    scene = json.loads(args.geometry_input.read_text())
    if (scene.get("status") != "native_scene_geometry_and_settling_checked"
            and not (scene.get("schema_version") == 1 and "robot_usd_sha256" in scene and "table_geometry" in scene)):
        raise ValueError("桌面检查需要原生场景报告")
    measured = ("velocity_measured_table", .002, True, True, 30., scene["table_geometry"])
    conditions = [measured] if args.measured_table_only else [*conditions, measured]
reports = []
for task in ("lift", "pick_place"):
    folder = root / f"outputs/rl_progress/{task}_body_physics_verified_50"
    metadata = json.loads((folder / "policy.json").read_text())
    trace_path = folder / "isaac_validation.hdf5"
    validation = json.loads((folder / "validation.json").read_text())
    if digest(trace_path) != validation["trace_sha256"]:
        raise ValueError("源轨迹 SHA256 与报告不一致")
    with h5py.File(trace_path, "r") as trace:
        names = set(name for component in COMPONENT_FIELDS.values() for name in component)
        names.update(("joint_position", "joint_velocity", "joint_targets", "joint_stiffness", "joint_damping", "joint_vel_limits",
                      "object_position_root", "object_quaternion_root", "object_linear_velocity_root", "object_angular_velocity_root", "terminated", "truncated"))
        fields = {name: trace[name][:] for name in names}
    for name, dt, servo, contact, effort, table in conditions:
        records, arrays = experiment(root / "outputs/so-arm100/Simulation/SO101/so101_old_calib.xml", metadata, fields, dt, servo, contact, effort, table)
        np.savez_compressed(args.output / f"{task}_{name}.npz", **arrays)
        result = {"task": task, "condition": name, "physics_dt": dt, "velocity_servo": servo,
                  "robot_contacts_enabled": contact, "effort_limit_nm": effort, "table_geometry": table,
                  "source_trace_sha256": digest(trace_path), "environments": records}
        reports.append(result)
        print(f"{task} {name}: RMSE={np.sqrt(np.mean([record['joint_position_rmse_rad']**2 for record in records])):.6f}, speed={max(record['maximum_speed_rad_s'] for record in records):.6f}", flush=True)
report = {"status": "actual_action_trace_dynamics_checked", "conditions": reports,
          "geometry_input_sha256": digest(args.geometry_input) if args.geometry_input else None,
          "source_code_sha256": digest(Path(__file__)), "physics_equivalence_verified": False, "task_success_verified": False}
(args.output / "dynamics_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
fig, axes = plt.subplots(3, 1, figsize=(13, 11), layout="constrained")
labels = [name.replace("velocity_", "v ").replace("position_", "p ") for name, *_ in conditions]
x = np.arange(len(conditions))
for task, offset, color in (("lift", -.18, "#2471a3"), ("pick_place", .18, "#d35400")):
    rows = [result for result in reports if result["task"] == task]
    values = [[1000*np.sqrt(np.mean([r["joint_position_rmse_rad"]**2 for r in row["environments"]])),
               max(r["maximum_speed_rad_s"] for r in row["environments"]),
               1000*max(r["maximum_object_error_m"] for r in row["environments"])] for row in rows]
    for index, axis in enumerate(axes):
        axis.bar(x+offset, np.array(values)[:, index], width=.34, label=task, color=color)
for axis, title in zip(axes, ("Joint position RMSE (mrad)", "Actual maximum joint speed (rad/s)", "Maximum object position difference (mm)"), strict=True):
    axis.set_title(title)
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.grid(axis="y", alpha=.25)
axes[0].legend()
axes[1].axhline(2, color="black", linestyle="--", label="Isaac configured joint velocity limit")
fig.suptitle("SO-101: actual Isaac action traces, recorded body physics, controlled MuJoCo conditions")
fig.savefig(args.output / "dynamics_conditions.png", dpi=160)
plt.close(fig)
