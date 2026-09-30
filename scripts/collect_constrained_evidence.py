import argparse
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("--local-run-id", required=True)
parser.add_argument("--linux-reports", type=Path, required=True)
parser.add_argument("--gripper", type=Path, required=True)
parser.add_argument("--linux-gripper", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--native-gripper", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[1]
records = []
provenance_fields = ("policy_sha256", "isaac_trace_sha256", "policy_metadata_sha256", "robot_model_sha256",
                     "robot_meshes", "constrained_drive_source_sha256", "recorded_physics_source_sha256", "collision_bundle_sha256")
for task in ("lift", "pick_place"):
    local = root / f"outputs/rl_progress/{task}_constrained_{args.local_run_id}_compare"
    report = json.loads((local / "report.json").read_text())
    linux_path = args.linux_reports / f"{task}_compare.json"
    linux = json.loads(linux_path.read_text())
    if any(report[field] != linux[field] for field in provenance_fields):
        raise ValueError("两台主机的实际策略、场景、物理代码或碰撞部件来源不一致")
    if digest(local / "comparison.hdf5") != report["trajectory_sha256"]:
        raise ValueError("配对轨迹 SHA256 检查失败")
    maximum_difference = 0.
    control_steps = physics_steps = 0
    maximum_speed = 0.
    with h5py.File(local / "comparison.hdf5", "r") as trace:
        for environment, (left, right) in enumerate(zip(report["environments"], linux["environments"], strict=True)):
            for field in ("joint_position_rmse_rad", "joint_position_max_error_rad", "joint_velocity_rmse_rad_s"):
                if not np.allclose(left[field], right[field], atol=1e-7, rtol=1e-7):
                    raise RuntimeError("两台主机的配对关节记录未通过一致性检查")
            group = trace[f"environment_{environment:06d}"]
            speed = group["physics_steps/joint_velocity"][:]
            limits = group["physics_steps/velocity_limits"][:]
            force = group["physics_steps/actuator_force"][:]
            if (not np.isfinite(speed).all() or not np.isfinite(force).all()
                    or np.any(np.abs(speed) > limits + 1e-8) or np.max(np.abs(force)) > 30 + 1e-8):
                raise RuntimeError("实际速度或力矩验收失败")
            difference = group["mujoco/joint_position"][:] - group["isaac/joint_position"][:]
            maximum_difference = max(maximum_difference, float(np.abs(difference).max()))
            control_steps += len(difference)
            physics_steps += len(speed)
            maximum_speed = max(maximum_speed, float(np.abs(speed).max()))
    evaluation_folder = root / f"outputs/rl_progress/{task}_constrained_{args.local_run_id}_mujoco"
    evaluation = json.loads((evaluation_folder / "report.json").read_text())
    linux_evaluation_path = args.linux_reports / f"{task}_mujoco.json"
    linux_evaluation = json.loads(linux_evaluation_path.read_text())
    if (any(evaluation[field] != linux_evaluation[field] for field in provenance_fields)
            or digest(evaluation_folder / "trajectory.hdf5") != evaluation["trajectory_sha256"]):
        raise ValueError("策略反馈评估来源检查失败")
    for host_evaluation in (evaluation, linux_evaluation):
        if not host_evaluation["actual_velocity_limits_verified"] or len(host_evaluation["episodes"]) != 4:
            raise RuntimeError("策略反馈评估的数量或速度检查失败")
    with h5py.File(evaluation_folder / "trajectory.hdf5", "r") as trace:
        for group in trace.values():
            velocity = group["physics_steps/joint_velocity"][:]
            force = group["physics_steps/actuator_force"][:]
            if not np.isfinite(velocity).all() or not np.isfinite(force).all() or np.max(np.abs(velocity)) > 2 + 1e-8 or np.max(np.abs(force)) > 30 + 1e-8:
                raise RuntimeError("策略反馈轨迹实际速度或力矩检查失败")
    records.append({"task": task, "paired_environments_per_host": len(report["environments"]),
                    "control_steps_per_host": control_steps, "paired_physics_steps_per_host": physics_steps,
                    "maximum_joint_position_difference_rad": maximum_difference,
                    "maximum_actual_joint_speed_rad_s": maximum_speed, "paired_host_metrics_verified": True,
                    "feedback_episodes_per_host": len(evaluation["episodes"]),
                    "feedback_success_rate_mac": evaluation["success_rate"],
                    "feedback_success_rate_linux": linux_evaluation["success_rate"],
                    "provenance": {field: report[field] for field in provenance_fields}})
    for name, path in (("compare", local / "report.json"), ("mujoco", evaluation_folder / "report.json"),
                       ("linux_compare", linux_path), ("linux_mujoco", linux_evaluation_path)):
        shutil.copyfile(path, args.output / f"{task}_{name}.json")
gripper_records = []
plans = []
for host, folder in (("mac", args.gripper), ("linux", args.linux_gripper)):
    report = json.loads((folder / "report.json").read_text())
    plan = json.loads((folder / "plan.json").read_text())
    plans.append(np.asarray(plan["joint_targets"]))
    if digest(folder / "trajectory.hdf5") != report["trajectory_sha256"] or digest(folder / "plan.json") != report["plan_sha256"]:
        raise ValueError("夹爪验收轨迹与计划 SHA256 检查失败")
    with h5py.File(folder / "trajectory.hdf5", "r") as trace:
        hold = trace["phase_index"][:] == 4
        height = trace["object_position_root"][:, 2] - plan["object_position_root"][2]
        forces = trace["jaw_forces"][:]
        velocity = trace["physics_steps/joint_velocity"][:]
        torque = trace["physics_steps/actuator_force"][:]
        if (not np.isfinite(height).all() or not np.isfinite(forces).all()
                or not np.isfinite(velocity).all() or not np.isfinite(torque).all()
                or np.max(np.abs(velocity)) > 2 + 1e-8 or np.max(np.abs(torque)) > 30 + 1e-8
                or hold.sum()*plan["control_dt"] < 1 - 1e-8
                or not np.all(height[hold] > .04) or not np.all(forces[hold] > .5)):
            raise RuntimeError("真实夹爪接触、持续提升、速度或力矩验收失败")
        gripper_records.append({"host": host, "physical_grasp_verified": True, "hold_duration_seconds": float(hold.sum()*plan["control_dt"]),
                                "minimum_hold_height_m": float(height[hold].min()), "maximum_height_m": float(height.max()),
                                "minimum_hold_jaw_forces_n": forces[hold].min(axis=0).tolist(),
                                "physics_steps": len(velocity), "source": report})
    shutil.copyfile(folder / "report.json", args.output / f"gripper_{host}.json")
plan_error = float(np.abs(plans[0] - plans[1]).max())
if plan_error > 1e-7:
    raise RuntimeError("两台主机的夹爪目标计划误差超过验收范围")
if any(gripper_records[0]["source"][field] != gripper_records[1]["source"][field]
       for field in ("source_code_sha256", "source_trace_sha256", "constrained_drive_source_sha256", "recorded_physics_source_sha256", "collision_bundle_sha256", "plan_sha256", "shared_plan_sha256")):
    raise ValueError("两台主机的夹爪输入或代码来源不一致")
result = {"status": "physical_drive_and_gripper_acceptance_verified", "tasks": records, "gripper": gripper_records,
          "maximum_cross_host_gripper_target_difference_rad": plan_error,
          "physics_equivalence_verified": False, "rl_policy_success_verified": False,
          "rl_training_started": False, "validation_source_sha256": digest(Path(__file__))}
if args.native_gripper:
    native = json.loads((args.native_gripper / "report.json").read_text())
    plan = json.loads((args.gripper / "plan.json").read_text())
    if (digest(args.native_gripper / "trajectory.hdf5") != native["trajectory_sha256"]
            or native["plan_sha256"] != gripper_records[0]["source"]["plan_sha256"]):
        raise ValueError("原生夹爪检查的轨迹或共享计划校验失败")
    with h5py.File(args.native_gripper / "trajectory.hdf5", "r") as trace:
        velocity = trace["physics_steps/joint_velocity"][:]
        height = trace["object_position_root"][:, :, 2] - plan["object_position_root"][2]
        forces = trace["jaw_forces"][:]
        hold = np.asarray(plan["phases"]) == "hold"
        if (not np.isfinite(velocity).all() or not np.isfinite(height).all() or not np.isfinite(forces).all()
                or height.shape != (native["control_steps"], len(native["environments"]))
                or len(velocity) != native["physics_steps_per_environment"]):
            raise RuntimeError("原生夹爪记录数量或数值检查失败")
        result["native_gripper"] = {"status": "native_contact_measured", "physics_dt": native["physics_dt"],
                                    "solver_velocity_iterations": native["solver_velocity_iterations"],
                                    "maximum_depenetration_velocity_m_s": native["maximum_depenetration_velocity_m_s"],
                                    "minimum_hold_height_m": height[hold].min(axis=0).tolist(),
                                    "minimum_hold_jaw_forces_n": forces[hold].min(axis=0).tolist(),
                                    "bilateral_lift_hold_verified": bool(np.all(height[hold] > .04) and np.all(forces[hold] > .5)),
                                    "maximum_joint_speed_rad_s": np.abs(velocity).max(axis=(0, 1)).tolist(),
                                    "expected_speed_limit_rad_s": 2.,
                                    "actual_velocity_limits_verified": bool(np.all(np.abs(velocity) <= 2 + 1e-8)),
                                    "trajectory_sha256": native["trajectory_sha256"], "source_code_sha256": native["source_code_sha256"],
                                    "plan_sha256": native["plan_sha256"], "physics_equivalence_verified": False}
    shutil.copyfile(args.native_gripper / "report.json", args.output / "native_gripper.json")
(args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": result["status"], "tasks": records, "gripper_target_difference_rad": plan_error}), flush=True)
