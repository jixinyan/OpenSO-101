import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from openso101.rl.config import digest
from openso101.sim2sim.mujoco import JOINT_NAMES, JOINT_OFFSETS, build_model


parser = argparse.ArgumentParser()
parser.add_argument("--states", type=Path, required=True)
parser.add_argument("--robot-model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--collision-bundle", type=Path)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
states = json.loads(args.states.read_text())
model = (build_model(args.robot_model, states["planner_physics"], args.collision_bundle) if args.collision_bundle
         else mujoco.MjModel.from_xml_path(str(args.robot_model)))
if tuple(model.joint(index).name for index in range(6)) != JOINT_NAMES:
    raise ValueError("任务规划需要官方 SO-101 old-calibration MJCF")
data = mujoco.MjData(model)
ids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
object_qpos = model.joint("object_free").qposadr[0] if args.collision_bundle else None
gripper = model.body("gripper").id
limits = np.asarray(states["soft_joint_limits"], dtype=float)[:5] + JOINT_OFFSETS[:5, None]
center = np.array([.01, 0., -.09])
rng = np.random.default_rng(states["seed"])
records = []
for environment in states["environments"]:
    previous = np.asarray(environment["joint_position"][:5]) + JOINT_OFFSETS[:5]
    start = np.asarray(environment["object_position_root"])
    goal = np.asarray(environment["goal_position_root"])
    if args.collision_bundle:
        data.qpos[object_qpos:object_qpos + 3] = start
        data.qpos[object_qpos + 3:object_qpos + 7] = [1., 0., 0., 0.]

    def penetration():
        depths = []
        for contact in data.contact:
            geometries = [model.geom(index).name for index in (contact.geom1, contact.geom2)]
            bodies = [model.body(model.geom_bodyid[index]).name for index in (contact.geom1, contact.geom2)]
            if "table" in geometries and set(bodies) <= {"world", "base", "object"}:
                continue
            depths.append(max(0., -float(contact.dist)))
        return max(depths, default=0.)

    targets = []
    for name, position, vertical in (("approach", start + [0., 0., .06], True),
                                     ("grasp", start, True), ("lift", goal, False)):
        def residual(arm):
            data.qpos[ids[:5]] = arm
            data.qpos[ids[5]] = .8
            mujoco.mj_forward(model, data)
            rotation = data.xmat[gripper].reshape(3, 3)
            error = 10 * (data.xpos[gripper] + rotation @ center - position)
            if args.collision_bundle:
                orientation = rotation[:, 2] - [0., 0., 1.] if vertical else np.zeros(3)
                return np.concatenate((error, orientation, [50 * penetration()]))
            inclination = max(0., np.cos(np.pi / 4) - rotation[2, 2])
            return np.concatenate((error, [inclination])) if vertical else error

        starts = [np.clip(previous, limits[:, 0] + 1e-5, limits[:, 1] - 1e-5)]
        starts.extend(rng.uniform(limits[:, 0] + 1e-5, limits[:, 1] - 1e-5, size=(15, 5)))
        solutions = [least_squares(residual, initial, bounds=(limits[:, 0] + 1e-5, limits[:, 1] - 1e-5),
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=500) for initial in starts]
        feasible = [item for item in solutions if item.success and np.linalg.norm(residual(item.x)[:3]) <= .03
                    and (not vertical or np.linalg.norm(residual(item.x)[3:-1] if args.collision_bundle
                                                        else residual(item.x)[3:]) <= .01)
                    and (not args.collision_bundle or residual(item.x)[-1] <= .005)]
        result = min(feasible, key=lambda item: np.linalg.norm(item.x - previous)) if feasible else min(
            solutions, key=lambda item: np.linalg.norm(residual(item.x)))
        errors = residual(result.x)
        position_error = float(np.linalg.norm(errors[:3]) / 10)
        orientation_error = float(np.linalg.norm(errors[3:-1] if args.collision_bundle else errors[3:])) if vertical else None
        penetration_m = float(errors[-1] / 50) if args.collision_bundle else None
        accepted = bool(result.success and position_error <= .003
                        and (orientation_error is None or orientation_error <= .01)
                        and (penetration_m is None or penetration_m <= .0001))
        targets.append({"phase": name, "target_position_root": position.tolist(),
                        "joint_position": (result.x - JOINT_OFFSETS[:5]).tolist(),
                        "position_error_m": position_error, "inclination_limit_violation": orientation_error,
                        "maximum_penetration_m": penetration_m,
                        "accepted": accepted, "attempts": len(solutions)})
        previous = result.x
    records.append({"environment": environment["environment"], "targets": targets,
                    "accepted": all(item["accepted"] for item in targets)})
report = {"status": "kinematic_plan_verified" if all(item["accepted"] for item in records) else "kinematic_plan_failed",
          "environments": records, "states_sha256": digest(args.states), "robot_model_sha256": digest(args.robot_model),
          "planner_source_sha256": digest(Path(__file__)), "maximum_grasp_inclination_rad": .01 if args.collision_bundle else float(np.pi / 4),
          "collision_bundle_sha256": digest(args.collision_bundle / "manifest.json") if args.collision_bundle else None,
          "waypoint_collision_verified": bool(args.collision_bundle) and all(item["accepted"] for item in records),
          "collision_path_verified": False,
          "task_success_verified": False}
with args.output.open("x") as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report), flush=True)
if report["status"] != "kinematic_plan_verified":
    raise RuntimeError("实际 reset 的任务目标未通过 IK 规划检查")
