import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from openso101.rl.config import digest
from openso101.sim2sim.grasp_planning import plan_collision_grasp
from openso101.sim2sim.mujoco import JOINT_NAMES, JOINT_OFFSETS, build_model


parser = argparse.ArgumentParser()
parser.add_argument("--states", type=Path, required=True)
parser.add_argument("--robot-model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--collision-bundle", type=Path)
sources = parser.add_mutually_exclusive_group()
sources.add_argument("--waypoint-seeds", type=Path)
sources.add_argument("--verified-base-plan", type=Path)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
states = json.loads(args.states.read_text())
seed_records = {}
base_records = {}
if args.verified_base_plan is not None:
    base = json.loads(args.verified_base_plan.read_text())
    if (not args.collision_bundle or base["status"] != "kinematic_plan_verified"
            or base["states_sha256"] != digest(args.states)
            or base["robot_model_sha256"] != digest(args.robot_model)
            or base["collision_bundle_sha256"] != digest(args.collision_bundle / "manifest.json")):
        raise ValueError("完整基础规划需要相同的实际环境、机器人与 collision bundle")
    base_records = {item["environment"]: item for item in base["environments"]}
    if set(base_records) != {item["environment"] for item in states["environments"]}:
        raise ValueError("完整基础规划的环境数量与实际环境不一致")
if args.waypoint_seeds is not None:
    seeds = json.loads(args.waypoint_seeds.read_text())
    if (not args.collision_bundle or seeds["states_sha256"] != digest(args.states)
            or seeds["robot_model_sha256"] != digest(args.robot_model)
            or seeds["collision_bundle_sha256"] != digest(args.collision_bundle / "manifest.json")):
        raise ValueError("waypoint 初始值需要相同的实际环境、机器人与 collision bundle")
    seed_records = {item["environment"]: [target["joint_position"] for target in item["targets"]
                                         if target["phase"] != "retreat"]
                    for item in seeds["environments"]}
model = (build_model(args.robot_model, states["planner_physics"], args.collision_bundle) if args.collision_bundle
         else mujoco.MjModel.from_xml_path(str(args.robot_model)))
if tuple(model.joint(index).name for index in range(6)) != JOINT_NAMES:
    raise ValueError("任务规划需要官方 SO-101 old-calibration MJCF")
data = mujoco.MjData(model)
ids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
gripper = model.body("gripper").id
limits = np.asarray(states["soft_joint_limits"], dtype=float)[:5] + JOINT_OFFSETS[:5, None]
center = np.array([.01, 0., -.09])
rng = np.random.default_rng(states["seed"])
records = []
waypoint_seed = None
for environment in states["environments"]:
    if args.collision_bundle:
        initial_waypoints = seed_records.get(environment["environment"], waypoint_seed)
        record = plan_collision_grasp(model, states, environment, rng, initial_waypoints,
                                      base_records.get(environment["environment"]))
        records.append(record)
        if record["accepted"]:
            waypoint_seed = [target["joint_position"] for target in record["targets"] if target["phase"] != "retreat"]
        continue
    previous = np.asarray(environment["joint_position"][:5]) + JOINT_OFFSETS[:5]
    start = np.asarray(environment["object_position_root"])
    goal = np.asarray(environment["goal_position_root"])
    targets = []
    specifications = [("approach", start + [0., 0., .06], True),
                      ("grasp", start, True), ("lift", goal, False)]
    for name, position, vertical in specifications:
        def residual(arm):
            data.qpos[ids[:5]] = arm
            data.qpos[ids[5]] = .8
            mujoco.mj_forward(model, data)
            rotation = data.xmat[gripper].reshape(3, 3)
            error = 10 * (data.xpos[gripper] + rotation @ center - position)
            inclination = max(0., np.cos(np.pi / 4) - rotation[2, 2])
            return np.concatenate((error, [inclination])) if vertical else error

        starts = [np.clip(previous, limits[:, 0] + 1e-5, limits[:, 1] - 1e-5)]
        starts.extend(rng.uniform(limits[:, 0] + 1e-5, limits[:, 1] - 1e-5, size=(15, 5)))
        solutions = [least_squares(residual, initial, bounds=(limits[:, 0] + 1e-5, limits[:, 1] - 1e-5),
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=500) for initial in starts]
        feasible = [item for item in solutions if item.success and np.linalg.norm(residual(item.x)[:3]) <= .03
                    and (not vertical or np.linalg.norm(residual(item.x)[3:]) <= .01)]
        result = min(feasible, key=lambda item: np.linalg.norm(item.x - previous)) if feasible else min(
            solutions, key=lambda item: np.linalg.norm(residual(item.x)))
        errors = residual(result.x)
        position_error = float(np.linalg.norm(errors[:3]) / 10)
        orientation_error = float(np.linalg.norm(errors[3:])) if vertical else None
        accepted = bool(result.success and position_error <= .003
                        and (orientation_error is None or orientation_error <= .01))
        targets.append({"phase": name, "target_position_root": position.tolist(),
                        "joint_position": (result.x - JOINT_OFFSETS[:5]).tolist(),
                        "position_error_m": position_error, "inclination_limit_violation": orientation_error,
                        "accepted": accepted, "attempts": len(solutions)})
        previous = result.x
    targets.sort(key=lambda target: ("approach", "grasp", "lift").index(target["phase"]))
    records.append({"environment": environment["environment"], "targets": targets,
                    "accepted": all(item["accepted"] for item in targets)})
report = {"status": "kinematic_plan_verified" if all(item["accepted"] for item in records) else "kinematic_plan_failed",
          "environments": records, "states_sha256": digest(args.states), "robot_model_sha256": digest(args.robot_model),
          "planner_source_sha256": digest(Path(__file__)), "maximum_grasp_inclination_rad": float(np.pi / 4),
          "waypoint_seed_sha256": digest(args.waypoint_seeds) if args.waypoint_seeds else None,
          "verified_base_plan_sha256": digest(args.verified_base_plan) if args.verified_base_plan else None,
          "collision_planner_sha256": digest(Path("src/openso101/sim2sim/grasp_planning.py")),
          "grasp_height_fraction_of_object": .25 if args.collision_bundle else 0.,
          "collision_bundle_sha256": digest(args.collision_bundle / "manifest.json") if args.collision_bundle else None,
          "waypoint_collision_verified": bool(args.collision_bundle) and all(item["accepted"] for item in records),
          "sampled_collision_path_verified": bool(args.collision_bundle) and all(item["accepted"] for item in records),
          "continuous_collision_path_verified": False,
          "task_success_verified": False}
if args.collision_bundle:
    report["maximum_grasp_inclination_rad"] = None
    report["orientation_constraint"] = (
        "approach_grasp_lift_rotation_difference_below_0.02_all_waypoints_and_cartesian_samples_tilt_below_pi_over_4"
        if states["task"] == "OpenSO101-PickPlace-v0" else "coupled_waypoints_rotation_matrix_difference_norm_below_0.02")
    report["held_object_collision_verified"] = states["task"] == "OpenSO101-PickPlace-v0" and all(
        item["accepted"] for item in records)
    report["held_object_pose_scope"] = "rigid_grasp_transform_for_kinematic_planning"
    report["native_task_goal_radius_m"] = states["planner_physics"]["task_goal_radius"]
with args.output.open("x") as stream:
    json.dump(report, stream, indent=2)
print(json.dumps({name: value for name, value in report.items() if name != "environments"}), flush=True)
if report["status"] != "kinematic_plan_verified":
    raise RuntimeError("实际 reset 的任务目标未通过 IK 规划检查")
