import argparse
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from openso101.rl.config import digest
from openso101.sim2sim.grasp_planning import forward_collision_geometry
from openso101.sim2sim.mujoco import JOINT_NAMES, JOINT_OFFSETS, build_model

import mujoco


parser = argparse.ArgumentParser()
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--robot-model", type=Path, required=True)
parser.add_argument("--collision-bundle", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
report = json.loads((args.native / "report.json").read_text())
states = json.loads((args.native / "initial_states.json").read_text())
if (digest(args.native / "trajectory.hdf5") != report["trace_sha256"]
        or digest(args.native / "initial_states.json") != report["states_sha256"]):
    raise ValueError("原生轨迹与状态的 SHA256 不一致")
model = build_model(args.robot_model, states["planner_physics"], args.collision_bundle)
reference, geometry = mujoco.MjData(model), mujoco.MjData(model)
joint_ids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
object_id = int(model.joint("object_free").qposadr[0])
maximum_pose_error, maximum_contact_error = 0., 0.
full_seconds, geometry_seconds, contacts, poses = 0., 0., 0, 0
with h5py.File(args.native / "trajectory.hdf5") as stream:
    joints = stream["joint_position"][:]
    positions = stream["object_position_root"][:]
    rotations = stream["object_quaternion_root"][:]
    active = stream["active"][:]
for step, environment in np.argwhere(active):
    for data in (reference, geometry):
        data.qpos[joint_ids] = joints[step, environment] + JOINT_OFFSETS
        data.qpos[object_id:object_id + 3] = positions[step, environment]
        data.qpos[object_id + 3:object_id + 7] = rotations[step, environment]
    start = perf_counter()
    mujoco.mj_forward(model, reference)
    full_seconds += perf_counter() - start
    start = perf_counter()
    forward_collision_geometry(model, geometry)
    geometry_seconds += perf_counter() - start
    if reference.ncon != geometry.ncon:
        raise ValueError("几何计算的实际 contact 数量不一致")
    error = max(float(np.abs(reference.xpos - geometry.xpos).max()),
                float(np.abs(reference.xmat - geometry.xmat).max()),
                float(np.abs(reference.geom_xpos - geometry.geom_xpos).max()),
                float(np.abs(reference.geom_xmat - geometry.geom_xmat).max()))
    maximum_pose_error = max(maximum_pose_error, error)
    for expected, actual in zip(reference.contact, geometry.contact, strict=True):
        if (expected.geom1, expected.geom2) != (actual.geom1, actual.geom2):
            raise ValueError("几何计算的实际 contact 实体不一致")
        maximum_contact_error = max(maximum_contact_error, abs(float(expected.dist - actual.dist)),
                                    float(np.abs(expected.pos - actual.pos).max()),
                                    float(np.abs(expected.frame - actual.frame).max()))
    if maximum_pose_error > 1e-12 or maximum_contact_error > 1e-12:
        raise ValueError("几何计算与完整 forward 的实际结果不一致")
    contacts += reference.ncon
    poses += 1
result = {"status": "actual_planner_geometry_verified", "poses": poses, "contacts": contacts,
          "maximum_pose_error": maximum_pose_error, "maximum_contact_error": maximum_contact_error,
          "full_forward_seconds": full_seconds, "collision_geometry_seconds": geometry_seconds,
          "speed_ratio": full_seconds / geometry_seconds, "mujoco_version": mujoco.__version__,
          "native_report_sha256": digest(args.native / "report.json"),
          "trajectory_sha256": report["trace_sha256"], "source_sha256": digest(Path(__file__)),
          "planner_sha256": digest(Path("src/openso101/sim2sim/grasp_planning.py")),
          "robot_model_sha256": digest(args.robot_model),
          "collision_bundle_sha256": digest(args.collision_bundle / "manifest.json"),
          "task_success_verified": False}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2), flush=True)
