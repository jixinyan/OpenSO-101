import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("--portable", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
metadata = json.loads((args.portable / "policy.json").read_text())
validation = json.loads((args.portable / "validation.json").read_text())
trajectory = args.portable / "isaac_validation.hdf5"
if metadata["task_profile"] != "grasp_v4" or digest(trajectory) != validation["trace_sha256"]:
    raise ValueError("检查需要完整范围控制与已校验的实际原生轨迹")
args.output.mkdir(parents=True, exist_ok=False)
records = []
fig, axes = plt.subplots(validation["num_envs"], 4, figsize=(17, 3 * validation["num_envs"]), squeeze=False)
with h5py.File(trajectory) as stream:
    steps = validation["validation_steps"]
    for index in range(validation["num_envs"]):
        positions = stream["object_position_root"][:, index]
        gripper_positions = stream["gripper_position_root"][:, index]
        quaternions = stream["gripper_quaternion_root"][:, index]
        rotations = Rotation.from_quat(quaternions, scalar_first=True)
        local = rotations.inv().apply(positions - gripper_positions)
        alignment = np.exp(-np.sum(((local - [.01, 0., -.09]) / [.04, .025, .03]) ** 2, axis=-1))
        inclination = np.arccos(np.clip(rotations.as_matrix()[:, 2, 2], -1, 1))
        forces = stream["jaw_forces"][:, index]
        angles = stream["joint_position"][:, index, -1]
        targets = stream["joint_targets"][:, index, -1]
        reached = stream["ee_object_distance"][:, index] < .08
        contact = (forces > metadata["grasp_force_threshold_newtons"]).all(axis=-1)
        above = positions[:, 2] > metadata["task_reference_height_root"] + .04
        resets = stream["terminated"][:, index] | stream["truncated"][:, index]
        times = np.arange(steps) * metadata["control_dt"]
        if not all(np.isfinite(value).all() for value in (local, alignment, inclination, angles, targets, forces)):
            raise ValueError("实际抓取记录包含无效数值")
        record = {"environment": index, "steps": steps, "reached_steps": int(reached.sum()),
                  "bilateral_contact_steps": int(contact.sum()), "held_above_table_steps": int((contact & above).sum()),
                  "jaw_target_range_rad": [float(targets.min()), float(targets.max())],
                  "jaw_actual_range_rad": [float(angles.min()), float(angles.max())],
                  "maximum_alignment": float(alignment.max()),
                  "gripper_inclination_range_rad": [float(inclination.min()), float(inclination.max())],
                  "completed_episodes": int(resets.sum()),
                  "other_contact_forces_recorded": "jaw_net_force_vectors" in stream,
                  "reached_local_position_median_m": np.median(local[reached], axis=0).tolist() if reached.any() else None,
                  "reached_alignment_median": float(np.median(alignment[reached])) if reached.any() else None,
                  "reached_inclination_median_rad": float(np.median(inclination[reached])) if reached.any() else None}
        if "jaw_net_force_vectors" in stream:
            other = stream["jaw_net_force_vectors"][:, index] - stream["jaw_object_force_vectors"][:, index]
            record["maximum_other_contact_force_n"] = np.linalg.norm(other, axis=-1).max(axis=0).tolist()
        axes[index, 0].plot(times, alignment, label="Actual grasp alignment")
        axes[index, 0].set(ylabel=f"Env {index} alignment")
        axes[index, 1].plot(times, targets, label="Commanded jaw")
        axes[index, 1].plot(times, angles, label="Actual jaw")
        axes[index, 1].set(ylabel="Jaw (rad)")
        axes[index, 2].plot(times, np.minimum(forces[:, 0], forces[:, 1]), label="Minimum bilateral force")
        axes[index, 2].set(ylabel="Contact force (N)")
        axes[index, 3].plot(times, positions[:, 2], label="Actual object height")
        axes[index, 3].set(ylabel="Height in robot root (m)")
        for axis in axes[index]:
            axis.set_xlabel("Actual control time (s)")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        records.append(record)
fig.suptitle("Actual saved RL policy · grasp geometry, commanded jaw and physical contact\n"
             "Recorded states span episode resets; task success uses independent evaluation")
fig.tight_layout()
fig.savefig(args.output / "policy_grasp.png", dpi=170)
plt.close(fig)
result = {"status": "actual_policy_grasp_diagnostics_recorded", "checkpoint_sha256": metadata["checkpoint_sha256"],
          "trace_sha256": digest(trajectory), "validation_sha256": digest(args.portable / "validation.json"),
          "metadata_sha256": digest(args.portable / "policy.json"), "source_sha256": digest(Path(__file__)),
          "environments": records, "figure_sha256": digest(args.output / "policy_grasp.png"),
          "rl_task_success_verified": False}
(args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
