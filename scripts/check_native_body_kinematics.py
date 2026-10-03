import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from openso101.rl.config import digest
from openso101.sim2sim.mujoco import JOINT_NAMES, JOINT_OFFSETS


parser = argparse.ArgumentParser()
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--robot-model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
report = json.loads((args.native / "report.json").read_text())
states = json.loads((args.native / "initial_states.json").read_text())
for name, field in (("trajectory.hdf5", "trace_sha256"), ("initial_states.json", "states_sha256"),
                    ("initial_physics.hdf5", "initial_physics_sha256")):
    if digest(args.native / name) != report[field]:
        raise ValueError(f"原生运动学来源 SHA256 不一致：{name}")
model = mujoco.MjModel.from_xml_path(str(args.robot_model))
data = mujoco.MjData(model)
joint_ids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
names = states["physics_recording"]["robot_body_names"]
body_ids = [model.body("moving_jaw_so101_v1" if name == "jaw" else name).id for name in names]
with h5py.File(args.native / "initial_physics.hdf5") as stream:
    initial = {name: stream[name][0] for name in (
        "joint_position", "robot_body_position_root", "robot_body_quaternion_root")}
with h5py.File(args.native / "trajectory.hdf5") as stream:
    trace = {name: stream[name][:] for name in (
        "joint_position", "robot_body_position_root", "robot_body_quaternion_root", "active", "phase")}
if any(not np.isfinite(value).all() for value in (*initial.values(), *trace.values())):
    raise ValueError("原生运动学来源包含无效数值")
records = []
for environment in range(report["episodes"]):
    data.qpos[joint_ids] = initial["joint_position"][environment] + JOINT_OFFSETS
    mujoco.mj_forward(model, data)
    rotations = data.xmat[body_ids].reshape(-1, 3, 3)
    native_rotations = Rotation.from_quat(initial["robot_body_quaternion_root"][environment], scalar_first=True).as_matrix()
    frame_rotation = rotations.swapaxes(-1, -2) @ native_rotations
    frame_translation = np.einsum("bij,bj->bi", rotations.swapaxes(-1, -2),
                                  initial["robot_body_position_root"][environment] - data.xpos[body_ids])
    indices = np.flatnonzero(trace["active"][:, environment] & (trace["phase"][:, environment] >= 0))
    if not len(indices):
        raise ValueError("运动学检查需要实际路径执行")
    position_errors, rotation_errors = [], []
    for step in indices:
        data.qpos[joint_ids] = trace["joint_position"][step, environment] + JOINT_OFFSETS
        mujoco.mj_forward(model, data)
        rotations = data.xmat[body_ids].reshape(-1, 3, 3)
        predicted_positions = data.xpos[body_ids] + np.einsum("bij,bj->bi", rotations, frame_translation)
        predicted_rotations = rotations @ frame_rotation
        actual_rotations = Rotation.from_quat(trace["robot_body_quaternion_root"][step, environment], scalar_first=True).as_matrix()
        position_errors.append(np.linalg.norm(predicted_positions - trace["robot_body_position_root"][step, environment], axis=-1))
        rotation_errors.append(Rotation.from_matrix(predicted_rotations.swapaxes(-1, -2) @ actual_rotations).magnitude())
    maximum_positions = np.asarray(position_errors).max(axis=0)
    maximum_rotations = np.asarray(rotation_errors).max(axis=0)
    records.append({"environment": environment, "control_steps": len(indices),
                    "jaw_angle_range_rad": [float(trace["joint_position"][indices, environment, 5].min()),
                                            float(trace["joint_position"][indices, environment, 5].max())],
                    "bodies": [{"body": name, "maximum_position_error_m": float(position),
                                "maximum_rotation_error_rad": float(rotation)}
                               for name, position, rotation in zip(names, maximum_positions, maximum_rotations, strict=True)],
                    "verified": bool((maximum_positions < 1e-5).all() and (maximum_rotations < 1e-4).all())})
result = {"status": "native_body_kinematics_verified" if all(record["verified"] for record in records)
          else "native_body_kinematics_failed", "environments": records,
          "position_tolerance_m": 1e-5, "rotation_tolerance_rad": 1e-4,
          "source_trace_sha256": report["trace_sha256"], "source_states_sha256": report["states_sha256"],
          "robot_model_sha256": digest(args.robot_model), "source_sha256": digest(Path(__file__)),
          "scope": "actual_native_body_poses_at_actual_joint_states_with_fixed_initial_frame_conversion",
          "contact_equivalence_verified": False}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
if result["status"] != "native_body_kinematics_verified":
    raise RuntimeError("实际原生实体运动未通过官方 MJCF 运动学检查")
