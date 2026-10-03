import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--replay", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
native_report = json.loads((args.native / "report.json").read_text())
replay_report = json.loads((args.replay / "report.json").read_text())
states = json.loads((args.native / "initial_states.json").read_text())
if digest(args.native / "report.json") != replay_report["source_report_sha256"]:
    raise ValueError("原生来源报告的 SHA256 不一致")
if digest(args.replay / "trajectory.hdf5") != replay_report["trajectory_sha256"]:
    raise ValueError("MuJoCo 轨迹的 SHA256 不一致")
if digest(args.native / "initial_states.json") != replay_report["source_states_sha256"]:
    raise ValueError("原生初始状态的 SHA256 不一致")
args.output.mkdir(parents=True, exist_ok=False)
parameters = states["task_parameters"]
fig, axes = plt.subplots(native_report["episodes"], 3, figsize=(15, 3.2 * native_report["episodes"]), squeeze=False)
records = []
with h5py.File(args.replay / "trajectory.hdf5") as trajectory:
    for index, environment in enumerate(states["environments"]):
        group = trajectory[f"environment_{index:06d}"]
        goal = np.asarray(environment["goal_position_root"])
        record = {"environment": index}
        for simulator, style in (("isaac", "-"), ("mujoco", "--")):
            positions = group[f"{simulator}/object_position_root"][:]
            forces = group[f"{simulator}/jaw_forces"][:]
            angles = group[f"{simulator}/joint_position"][:, 5]
            times = (np.arange(len(positions)) + 1) * states["control_dt"]
            height = positions[:, 2] > states["task_reference_height_root"] + parameters["minimal_height"]
            distance = np.linalg.norm(positions - goal, axis=-1)
            contact = (forces > parameters["force_threshold"]).all(axis=-1)
            eligible = height & (distance < parameters["goal_radius"]) & contact
            consecutive = np.zeros(len(eligible), dtype=int)
            for step, value in enumerate(eligible):
                consecutive[step] = (consecutive[step - 1] if step else 0) + 1 if value else 0
            record[simulator] = {
                "height_steps": int(height.sum()), "goal_steps": int((distance < parameters["goal_radius"]).sum()),
                "bilateral_contact_steps": int(contact.sum()), "eligible_steps": int(eligible.sum()),
                "longest_hold_seconds": float(consecutive.max() * states["control_dt"]),
                "final_hold_seconds": float(consecutive[-1] * states["control_dt"]),
                "final_goal_distance_m": float(distance[-1]), "final_jaw_angle_rad": float(angles[-1]),
                "final_contact_forces_n": forces[-1].tolist(),
            }
            axes[index, 0].plot(times, positions[:, 2], style, label=simulator)
            axes[index, 1].plot(times, angles, style, label=simulator)
            axes[index, 2].plot(times, consecutive * states["control_dt"], style, label=simulator)
        axes[index, 0].axhline(goal[2] - parameters["goal_radius"], color="black", alpha=.3)
        axes[index, 0].set(ylabel=f"Env {index} height (m)")
        axes[index, 1].set(ylabel="Jaw angle (rad)")
        axes[index, 2].axhline(parameters["settle_seconds"], color="black", alpha=.3, label="Required hold")
        axes[index, 2].set(ylabel="Eligible hold (s)")
        for axis in axes[index]:
            axis.set_xlabel("Time since settled initial state (s)")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        records.append(record)
fig.suptitle("Actual native joint-target replay: height, jaw contact and task hold")
fig.tight_layout()
fig.savefig(args.output / "native_mujoco_replay.png", dpi=170)
plt.close(fig)
result = {"environments": records, "task_parameters": parameters,
          "native_report_sha256": digest(args.native / "report.json"),
          "replay_report_sha256": digest(args.replay / "report.json"),
          "trajectory_sha256": replay_report["trajectory_sha256"], "source_sha256": digest(Path(__file__)),
          "figure_sha256": digest(args.output / "native_mujoco_replay.png")}
(args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
