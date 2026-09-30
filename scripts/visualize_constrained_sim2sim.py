import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from openso101.rl.config import digest
from openso101.sim2sim.mujoco import JOINT_NAMES


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--gripper", type=Path)
parser.add_argument("--native-gripper", type=Path)
parser.add_argument("--local-run-id", default="mac_bvh_verified")
parser.add_argument("--linux-reports", type=Path, required=True)
parser.add_argument("--baseline-gripper", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[1]
evidence = []
metrics = []
fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
colors = ("#2471a3", "#d35400")
for row, task in enumerate(("lift", "pick_place")):
    folder = root / f"outputs/rl_progress/{task}_constrained_{args.local_run_id}_compare"
    report = json.loads((folder / "report.json").read_text())
    linux = json.loads((args.linux_reports / f"{task}_compare.json").read_text())
    if (report["policy_sha256"] != linux["policy_sha256"] or report["isaac_trace_sha256"] != linux["isaac_trace_sha256"]
            or report["policy_metadata_sha256"] != linux["policy_metadata_sha256"]
            or report["constrained_drive_source_sha256"] != linux["constrained_drive_source_sha256"]
            or report["recorded_physics_source_sha256"] != linux["recorded_physics_source_sha256"]
            or report["collision_bundle_sha256"] != linux["collision_bundle_sha256"]):
        raise ValueError("两台主机的策略、源记录、场景或控制器不一致")
    for local_record, linux_record in zip(report["environments"], linux["environments"], strict=True):
        for name in ("joint_position_rmse_rad", "joint_position_max_error_rad", "joint_velocity_rmse_rad_s"):
            if not np.allclose(local_record[name], linux_record[name], atol=1e-7, rtol=1e-7):
                raise RuntimeError("两台主机的配对物理结果不一致")
    trace_path = folder / "comparison.hdf5"
    if digest(trace_path) != report["trajectory_sha256"]:
        raise ValueError("配对轨迹校验失败")
    joint_errors = []
    maximum_speed = 0.
    physical_steps = 0
    with h5py.File(trace_path, "r") as trace:
        for index, record in enumerate(report["environments"]):
            group = trace[f"environment_{index:06d}"]
            error = group["mujoco/joint_position"][:] - group["isaac/joint_position"][:]
            joint_errors.append(error)
            velocity = group["physics_steps/joint_velocity"][:]
            limits = group["physics_steps/velocity_limits"][:]
            force = group["physics_steps/actuator_force"][:]
            if np.any(np.abs(velocity) > limits + 1e-8) or np.max(np.abs(force)) > 30 + 1e-8:
                raise RuntimeError("实际速度或力矩记录未通过范围检查")
            physical_steps += len(velocity)
            maximum_speed = max(maximum_speed, float(np.max(np.abs(velocity))))
            if index == 0:
                time = np.arange(len(velocity)) * report["physics_dt"]
                axes[row, 0].plot(time, velocity[:, 1], label="MuJoCo constrained drive", color=colors[row])
                baseline = np.load(root / f"outputs/rl_progress/dynamics_debug/{task}_velocity_dt_0.002.npz")
                axes[row, 0].plot(time, baseline["env_0_physics_velocity"][:, 1], label="Velocity reference servo", color="#888888", alpha=.8)
                source_time = np.arange(record["steps"]) * report["control_dt"]
                axes[row, 0].plot(source_time, group["isaac/joint_velocity"][:, 1], label="Isaac measured", color="#239b56", linestyle="--")
        errors = np.concatenate(joint_errors)
        axes[row, 1].bar(np.arange(6), np.max(np.abs(errors), axis=0)*1000, color=colors[row])
        axes[row, 1].set_xticks(np.arange(6), JOINT_NAMES, rotation=25, ha="right")
        metrics.append({"task": task, "control_samples": len(errors), "physics_steps": physical_steps,
                        "maximum_position_difference_rad": float(np.abs(errors).max()),
                        "joint_position_rmse_rad": float(np.sqrt(np.mean(errors**2))),
                        "maximum_actual_speed_rad_s": maximum_speed, "paired_host_metrics_verified": True})
    axes[row, 0].axhline(2, color="black", linestyle=":")
    axes[row, 0].axhline(-2, color="black", linestyle=":")
    axes[row, 0].set_xlim(0, .35)
    axes[row, 0].set_title(f"{task}: shoulder_lift actual velocity")
    axes[row, 0].set_ylabel("rad/s")
    axes[row, 0].set_xlabel("Time (s)")
    axes[row, 1].set_title(f"{task}: maximum joint position difference")
    axes[row, 1].set_ylabel("mrad")
    for axis in axes[row]:
        axis.grid(alpha=.2)
    evidence.append({"file": str(trace_path.relative_to(root)), "sha256": digest(trace_path)})
axes[0, 0].legend(fontsize=8)
fig.suptitle("SO-101: actual source actions, recorded physics, physical velocity constraints")
fig.savefig(args.output / "constrained_drive.png", dpi=160)
plt.close(fig)
if args.gripper:
    if args.baseline_gripper is None:
        raise ValueError("夹爪图表需要同一控制器的基础碰撞记录")
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), layout="constrained")
    cases = [("MuJoCo upstream convex", args.baseline_gripper, "#888888"),
             ("MuJoCo CoACD parts", args.gripper, "#2471a3")]
    for label, folder, color in cases:
        report = json.loads((folder / "report.json").read_text())
        trace_path = folder / "trajectory.hdf5"
        if digest(trace_path) != report["trajectory_sha256"]:
            raise ValueError("夹爪物理轨迹校验失败")
        plan = json.loads((folder / "plan.json").read_text())
        with h5py.File(trace_path, "r") as trace:
            time = (np.arange(len(trace["joint_position"])) + 1) * plan["control_dt"]
            axes[0].plot(time, 1000*(trace["object_position_root"][:, 2]-plan["object_position_root"][2]), label=label, color=color)
            axes[1].plot(time, np.min(trace["jaw_forces"][:], axis=-1), label=label, color=color)
            axes[2].plot(time, trace["joint_position"][:, -1], label=label, color=color)
            velocity = trace["physics_steps/joint_velocity"][:]
            physics_time = (np.arange(len(velocity))+1)*plan["control_dt"]*len(time)/len(velocity)
            axes[3].plot(physics_time, np.abs(velocity).max(axis=-1), label=label, color=color)
        evidence.append({"file": str(trace_path), "sha256": digest(trace_path)})
    if args.native_gripper:
        report = json.loads((args.native_gripper / "report.json").read_text())
        trace_path = args.native_gripper / "trajectory.hdf5"
        if digest(trace_path) != report["trajectory_sha256"]:
            raise ValueError("原生夹爪物理轨迹校验失败")
        with h5py.File(trace_path, "r") as trace:
            time = (np.arange(report["control_steps"]) + 1) * report["control_dt"]
            for environment in range(len(report["environments"])):
                label = "Isaac SDF (4 environments)" if environment == 0 else None
                axes[0].plot(time, 1000*(trace["object_position_root"][:, environment, 2]-plan["object_position_root"][2]), label=label, color="#239b56", alpha=.6)
                axes[1].plot(time, np.min(trace["jaw_forces"][:, environment], axis=-1), label=label, color="#239b56", alpha=.6)
                axes[2].plot(time, trace["joint_position"][:, environment, -1], label=label, color="#239b56", alpha=.6)
                velocity = trace["physics_steps/joint_velocity"][:, environment]
                physics_time = (np.arange(len(velocity))+1)*report["physics_dt"]
                axes[3].plot(physics_time, np.abs(velocity).max(axis=-1), label=label, color="#239b56", alpha=.6)
        evidence.append({"file": str(trace_path), "sha256": digest(trace_path)})
    for axis, title in zip(axes, ("Object height above initial center (mm)", "Minimum of two jaw contact forces (N)", "Actual jaw position (rad)", "Maximum actual joint speed at each physics step (rad/s)"), strict=True):
        axis.set_title(title)
        axis.set_xlabel("Time (s)")
        axis.grid(alpha=.2)
    axes[0].axhline(40, color="black", linestyle=":")
    axes[1].axhline(.5, color="black", linestyle=":")
    axes[3].axhline(2, color="black", linestyle=":")
    axes[0].legend()
    fig.suptitle("Scripted grasp: shared targets, measured contact and actual velocity\nNative physical parameters recorded independently")
    fig.savefig(args.output / "gripper_mechanics.png", dpi=160)
    plt.close(fig)
images = {}
for path in args.output.glob("*.png"):
    with Image.open(path) as frame:
        frame.load()
        if frame.width < 1000 or frame.height < 800:
            raise RuntimeError("图表尺寸检查失败")
        images[path.name] = {"sha256": digest(path), "size": list(frame.size)}
(args.output / "report.json").write_text(json.dumps({"status": "physical_trace_visualizations_checked", "metrics": metrics,
                                                     "evidence": evidence, "images": images}, indent=2) + "\n")
print(json.dumps(metrics), flush=True)
