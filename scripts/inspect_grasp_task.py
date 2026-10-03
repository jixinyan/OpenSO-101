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
        valid = np.flatnonzero(trace["active"][:, environment] & (trace["phase"][:, environment] >= 0))
        phases = trace["phase"][valid, environment]
        target_indices = (phases - (phases >= 2)).clip(0, len(item["targets"]) - 1)
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
        if "path_cursor" in trace:
            records[-1]["path_progress"] = [{
                "phase": int(phase), "control_steps": int((phases == phase).sum()),
                "last_path_cursor": int(trace["path_cursor"][valid[phases == phase][-1], environment]),
                "last_command_error_rad": float(np.linalg.norm(
                    trace["desired_joint_position"][valid[phases == phase][-1], environment, :5]
                    - trace["joint_position"][valid[phases == phase][-1], environment, :5])),
            } for phase in np.unique(phases)]
        if "gravity_compensation" in trace:
            records[-1]["final_gravity_compensation_nm"] = trace["gravity_compensation"][valid[-1], environment].tolist()
            records[-1]["final_drive_target_difference_rad"] = (
                trace["joint_targets"][valid[-1], environment] - qpos[-1]).tolist()
        if "jaw_net_force_vectors" in trace:
            net = trace["jaw_net_force_vectors"][valid, environment]
            filtered = trace["jaw_object_force_vectors"][valid, environment]
            records[-1]["contact_diagnostics"] = {
                "maximum_net_force_n": np.linalg.norm(net, axis=-1).max(axis=0).tolist(),
                "maximum_other_contact_force_n": np.linalg.norm(net - filtered, axis=-1).max(axis=0).tolist(),
                "final_other_contact_force_n": np.linalg.norm(net[-1] - filtered[-1], axis=-1).tolist(),
                "scope": "all_contacts_minus_object_filtered_contacts",
            }
print(json.dumps(records, indent=2), flush=True)
