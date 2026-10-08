import argparse
import json
import os
from pathlib import Path

import h5py
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from openso101.rl.config import digest
from openso101.robots.so101.ik import differential_ik
from openso101.sim2sim.mujoco import JOINT_NAMES, JOINT_OFFSETS


parser = argparse.ArgumentParser()
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--robot-model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
    raise ValueError("键盘运动学检查需要禁止 CUDA，并使用新的报告文件")
source = json.loads((args.native / "report.json").read_text())
trace = args.native / "trajectory.hdf5"
if digest(trace) != source["trace_sha256"]:
    raise ValueError("实际机器人轨迹 SHA256 不一致")
with h5py.File(trace) as stream:
    positions = stream["joint_position"][:][stream["active"][:].astype(bool)]
if not len(positions) or positions.shape[1] != 6 or not np.isfinite(positions).all():
    raise ValueError("键盘运动学检查需要实际六关节姿态")
positions = positions[np.linspace(0, len(positions) - 1, min(len(positions), 64)).astype(int)]
model = mujoco.MjModel.from_xml_path(str(args.robot_model))
data = mujoco.MjData(model)
body = model.body("gripper").id
qpos_ids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
dof_ids = [int(model.joint(name).dofadr[0]) for name in JOINT_NAMES[:5]]
limits = torch.from_numpy((model.jnt_range[:5] - JOINT_OFFSETS[:5, None])[None])
dt = source["control_dt"]
directions = np.eye(4) * np.array([.001, .001, .001, .002])
commands = np.concatenate((directions, -directions))
records = []
for pose_index, position in enumerate(positions):
    data.qpos[qpos_ids] = position + JOINT_OFFSETS
    mujoco.mj_forward(model, data)
    old_rotation = data.xmat[body].reshape(3, 3).copy()
    old_point = data.xpos[body] + old_rotation @ np.array([.01, 0, -.09])
    linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, linear, angular, old_point, body)
    jacobian = torch.from_numpy(np.vstack((linear[:, dof_ids], angular[2:3, dof_ids]))[None])
    for command in commands:
        target = differential_ik(jacobian, torch.from_numpy(command[None]),
                                 torch.from_numpy(position[None, :5]), limits, dt=dt).numpy()[0]
        if np.abs(target - position[:5]).max() > dt + 1e-8:
            raise RuntimeError("键盘 IK 输出超过指定的实际关节速度范围")
        data.qpos[qpos_ids[:5]] = target + JOINT_OFFSETS[:5]
        data.qpos[qpos_ids[5]] = position[5]
        mujoco.mj_forward(model, data)
        rotation = data.xmat[body].reshape(3, 3)
        point = data.xpos[body] + rotation @ np.array([.01, 0, -.09])
        achieved = np.concatenate((point - old_point, [Rotation.from_matrix(rotation @ old_rotation.T).as_rotvec()[2]]))
        remaining = float(np.linalg.norm(command - achieved))
        initial = float(np.linalg.norm(command))
        records.append({"pose_index": pose_index, "command": command.tolist(), "target": target.tolist(),
                        "initial_error": initial, "remaining_error": remaining,
                        "maximum_joint_increment_rad": float(np.abs(target - position[:5]).max()),
                        "improved": remaining <= initial + 1e-7})
        data.qpos[qpos_ids] = position + JOINT_OFFSETS
        mujoco.mj_forward(model, data)
if not all(item["improved"] for item in records):
    raise RuntimeError("实际 MJCF 姿态的键盘 IK 增大了指定目标误差")
report = {"status": "actual_mjcf_keyboard_ik_verified", "poses": len(positions), "commands": len(records),
          "scope": "forward_kinematics_at_actual_recorded_joint_positions", "records": records,
          "source_trace_sha256": source["trace_sha256"], "robot_model_sha256": digest(args.robot_model),
          "source_sha256": digest(Path(__file__)), "ik_source_sha256": digest(Path("src/openso101/robots/so101/ik.py")),
          "gpu_tests_started": False, "physics_keyboard_collection_verified": False,
          "human_teleoperation_success_verified": False}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key != "records"}))
