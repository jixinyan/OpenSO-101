import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
args = parser.parse_args()
report = json.loads((args.folder / "report.json").read_text())
plan = json.loads((args.folder / "plan.json").read_text())
if digest(args.folder / "trajectory.hdf5") != report["trace_sha256"]:
    raise ValueError("任务轨迹的 SHA256 检查失败")
records = []
with h5py.File(args.folder / "trajectory.hdf5") as trace:
    for environment, item in enumerate(plan["environments"]):
        valid = np.flatnonzero(trace["active"][:, environment])
        phases = trace["phase"][valid, environment]
        target_indices = np.where(phases == 2, 1, phases).clip(0, 2)
        target_joints = np.asarray([entry["joint_position"] for entry in item["targets"]])[target_indices]
        target_positions = np.asarray([entry["target_position_root"] for entry in item["targets"]])[target_indices]
        qpos = trace["joint_position"][valid, environment]
        grip = trace["grasp_position_root"][valid, environment]
        positions = trace["object_position_root"][valid, environment]
        records.append({"environment": environment, "final_phase": int(phases[-1]),
                        "final_joint_error_rad": (target_joints[-1] - qpos[-1, :5]).tolist(),
                        "minimum_grasp_target_error_m": float(np.linalg.norm(grip - target_positions, axis=-1).min()),
                        "final_grasp_target_error_m": float(np.linalg.norm(grip[-1] - target_positions[-1])),
                        "minimum_ee_object_distance_m": float(np.linalg.norm(grip - positions, axis=-1).min()),
                        "final_ee_object_distance_m": float(np.linalg.norm(grip[-1] - positions[-1])),
                        "final_joint_velocity_rad_s": trace["joint_velocity"][valid[-1], environment].tolist(),
                        "final_object_position_root_m": positions[-1].tolist(),
                        "final_grasp_position_root_m": grip[-1].tolist()})
        if "gravity_compensation" in trace:
            records[-1]["final_gravity_compensation_nm"] = trace["gravity_compensation"][valid[-1], environment].tolist()
            records[-1]["final_drive_target_difference_rad"] = (
                trace["joint_targets"][valid[-1], environment] - qpos[-1]).tolist()
print(json.dumps(records, indent=2), flush=True)
