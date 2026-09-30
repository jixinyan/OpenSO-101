import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("--dynamics", type=Path, required=True)
parser.add_argument("--table", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
args.output.mkdir(parents=True, exist_ok=False)
dynamics = json.loads((args.dynamics / "dynamics_report.json").read_text())
table = json.loads((args.table / "dynamics_report.json").read_text())
fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
fig_table, table_axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
fig_policy, policy_axes = plt.subplots(2, 2, figsize=(13, 7), layout="constrained")
checks = []
for column, task in enumerate(("lift", "pick_place")):
    baseline = next(row for row in dynamics["conditions"] if row["task"] == task and row["condition"] == "velocity_dt_0.002")
    no_contact = next(row for row in dynamics["conditions"] if row["task"] == task and row["condition"] == "velocity_no_contact")
    effort = next(row for row in dynamics["conditions"] if row["task"] == task and row["condition"] == "velocity_effort_3.35")
    environment = max(baseline["environments"], key=lambda row: row["maximum_speed_rad_s"])["environment"]
    joint = 1
    original = np.load(args.dynamics / f"{task}_velocity_dt_0.002.npz")
    disabled = np.load(args.dynamics / f"{task}_velocity_no_contact.npz")
    limited_effort = np.load(args.dynamics / f"{task}_velocity_effort_3.35.npz")
    without_contact_error = max(float(np.max(np.abs(original[name]-disabled[name]))) for name in original.files if name.endswith(("joint_position", "physics_velocity", "physics_torque")))
    effort_error = max(float(np.max(np.abs(original[name]-limited_effort[name]))) for name in original.files)
    if without_contact_error > 1e-12 or effort_error > 1e-12:
        raise RuntimeError("接触或力矩条件未通过配对数据检查")
    folder = root / f"outputs/rl_progress/{task}_body_physics_verified_50"
    with h5py.File(folder / "isaac_validation.hdf5", "r") as trace:
        native_q = trace["joint_position"][:, environment, joint]
        native_v = trace["joint_velocity"][:, environment, joint]
        native_object = trace["object_position_root"][:, environment]
        native_objects = trace["object_position_root"][:]
        jaw = trace["joint_targets"][:, :, 5]
        raw_jaw = trace["raw_action"][:, :, 5]
        policy_time = np.arange(len(jaw))*.02
        for index in range(jaw.shape[1]):
            policy_axes[0, column].plot(policy_time, raw_jaw[:, index], label=f"environment {index}", alpha=.7)
            policy_axes[1, column].plot(policy_time, jaw[:, index], alpha=.7)
        policy_axes[0, column].set_title(f"{task}: actual Isaac raw gripper actions")
        policy_axes[0, column].axhline(1, color="black", linestyle="--", label="0.8 rad clipping boundary")
        policy_axes[1, column].set_title("Actual processed gripper targets (rad)")
        policy_axes[1, column].set_ylim(-.05, .85)
        policy_axes[1, column].set_xlabel("time (s)")
    for condition, label, color in (("velocity_dt_0.002", "MuJoCo velocity target, dt=2 ms", "#2471a3"),
                                    ("velocity_dt_0.0005", "MuJoCo velocity target, dt=0.5 ms", "#d35400")):
        arrays = np.load(args.dynamics / f"{task}_{condition}.npz")
        dt = next(row["physics_dt"] for row in dynamics["conditions"] if row["task"] == task and row["condition"] == condition)
        q = arrays[f"env_{environment}_joint_position"][:, joint]
        v = arrays[f"env_{environment}_physics_velocity"][:, joint]
        axes[0, column].plot((np.arange(len(v))+1)*dt, v, label=label, color=color)
        axes[1, column].plot(np.arange(len(q))*.02, 1000*(q-native_q[:len(q)]), color=color)
    axes[0, column].plot(np.arange(len(native_v))*.02, native_v, "o-", markersize=3, label="Isaac control samples", color="#148f77")
    axes[0, column].axhline(-2, color="black", linestyle="--", label="configured speed bound")
    axes[0, column].axhline(2, color="black", linestyle="--")
    axes[0, column].set_xlim(0, .25)
    axes[0, column].set_title(f"{task}, environment {environment}, shoulder_lift velocity (rad/s)")
    axes[1, column].set_title("Joint position difference (mrad)")
    axes[1, column].set_xlabel("time (s)")
    for folder_path, name, label, color in ((args.dynamics, "velocity_dt_0.002", "nominal table plane", "#d35400"),
                                          (args.table, "velocity_measured_table", "measured native table box", "#2471a3")):
        arrays = np.load(folder_path / f"{task}_{name}.npz")
        objects = arrays[f"env_{environment}_object_position"]
        table_axes[column].plot(np.arange(len(objects))*.02, 1000*(objects[:, 2]-native_object[:len(objects), 2]), label=label, color=color)
    table_axes[column].set_title(f"{task}: object vertical difference (mm)")
    table_axes[column].set_xlabel("time (s)")
    measured = next(row for row in table["conditions"] if row["task"] == task)
    measured_arrays = np.load(args.table / f"{task}_velocity_measured_table.npz")
    settled_error = max(float(np.linalg.norm(measured_arrays[f"env_{index}_object_position"][-1]
                                            - native_objects[len(original[f"env_{index}_object_position"])-1, index]))
                        for index in range(native_objects.shape[1]))
    checks.append({"task": task, "paired_no_contact_maximum_difference": without_contact_error,
                   "paired_effort_limit_maximum_difference": effort_error,
                   "baseline_maximum_object_error_m": max(row["maximum_object_error_m"] for row in baseline["environments"]),
                   "measured_table_maximum_object_error_m": max(row["maximum_object_error_m"] for row in measured["environments"]),
                   "measured_table_final_object_error_m": settled_error,
                   "raw_gripper_action_minimum": float(raw_jaw.min()), "raw_gripper_action_maximum": float(raw_jaw.max()),
                   "processed_gripper_target_minimum_rad": float(jaw.min()), "processed_gripper_target_maximum_rad": float(jaw.max())})
axes[0, 0].legend(fontsize=8)
table_axes[0].legend()
policy_axes[0, 0].legend(fontsize=8)
for figure in (fig, fig_table, fig_policy):
    for axis in figure.axes:
        axis.grid(alpha=.2)
for figure, name in ((fig, "velocity_constraint.png"), (fig_table, "table_geometry.png"), (fig_policy, "policy_gripper.png")):
    figure.savefig(args.output / name, dpi=160)
    plt.close(figure)
report = {"checks": checks, "dynamics_report_sha256": digest(args.dynamics / "dynamics_report.json"),
          "table_report_sha256": digest(args.table / "dynamics_report.json"),
          "source_code_sha256": digest(Path(__file__)),
          "figures": {path.name: digest(path) for path in sorted(args.output.glob("*.png"))}}
(args.output / "visualization_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(checks, indent=2))
