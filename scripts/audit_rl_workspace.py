import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.optimize import least_squares

from openso101.rl.config import digest
from openso101.sim2sim.mujoco import JOINT_NAMES, JOINT_OFFSETS

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=Path, required=True)
parser.add_argument("--robot-model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
metadata = json.loads((args.policy / "policy.json").read_text())
model = mujoco.MjModel.from_xml_path(str(args.robot_model))
data = mujoco.MjData(model)
qids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES[:5]]
bounds = model.jnt_range[[model.joint(name).id for name in JOINT_NAMES[:5]]]
gripper_id = model.body("gripper").id
with h5py.File(args.policy / "isaac_validation.hdf5") as trace:
    points = [(kind, env, trace[name][0, env, :3]) for env in range(trace["object_position_root"].shape[1])
              for kind, name in (("grasp", "object_position_root"), ("goal", "goal_root"))]
records = []
rng = np.random.default_rng(42)
for kind, environment, position in points:
    def residual(q):
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        rotation = data.xmat[gripper_id].reshape(3, 3)
        center = data.xpos[gripper_id] + rotation @ np.array([.01, 0, -.09])
        return np.concatenate((10 * (center - position), rotation[:, 2] - [0, 0, 1]))

    solutions = []
    initial = np.asarray(metadata["default_joint_positions"])[:5] + JOINT_OFFSETS[:5]
    for attempt in range(8):
        start = initial if attempt == 0 else rng.uniform(bounds[:, 0] + 1e-4, bounds[:, 1] - 1e-4)
        solution = least_squares(residual, start, bounds=(bounds[:, 0] + 1e-4, bounds[:, 1] - 1e-4), max_nfev=500)
        solutions.append((np.linalg.norm(residual(solution.x)), solution))
    error, best = min(solutions, key=lambda item: item[0])
    residual_value = residual(best.x)
    position_solutions = []
    for attempt in range(8):
        start = initial if attempt == 0 else rng.uniform(bounds[:, 0] + 1e-4, bounds[:, 1] - 1e-4)
        candidate = least_squares(lambda q: residual(q)[:3], start,
                                  bounds=(bounds[:, 0] + 1e-4, bounds[:, 1] - 1e-4), max_nfev=500)
        position_solutions.append((np.linalg.norm(residual(candidate.x)[:3]) / 10, candidate))
    position_error, position_solution = min(position_solutions, key=lambda item: item[0])
    records.append({"kind": kind, "environment": environment, "position_root": position.tolist(),
                    "position_error_m": float(np.linalg.norm(residual_value[:3]) / 10),
                    "orientation_axis_error": float(np.linalg.norm(residual_value[3:])),
                    "pose_reachable": bool(error < .01),
                    "position_only_error_m": float(position_error),
                    "position_only_reachable": bool(position_error < .001),
                    "joint_solution_isaac": (best.x - JOINT_OFFSETS[:5]).tolist()})
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({"task": metadata["task_id"], "records": records,
                                  "robot_sha256": digest(args.robot_model),
                                  "source_trace_sha256": digest(args.policy / "isaac_validation.hdf5"),
                                  "method": "bounded_multistart_IK_with_vertical_gripper_axis",
                                  "collision_path_verified": False}, indent=2) + "\n")
print(json.dumps(records), flush=True)
