import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from openso101.rl.config import digest
from openso101.sim2sim.mujoco import JOINT_NAMES, JOINT_OFFSETS, build_model


parser = argparse.ArgumentParser()
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--robot-model", type=Path, required=True)
parser.add_argument("--collision-bundle", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--stride", type=int, default=10)
args = parser.parse_args()
if args.output.exists() or args.stride <= 0:
    raise ValueError("接触几何检查需要新的输出文件与有效采样间隔")
report = json.loads((args.native / "report.json").read_text())
states = json.loads((args.native / "initial_states.json").read_text())
if digest(args.native / "trajectory.hdf5") != report["trace_sha256"]:
    raise ValueError("原生接触轨迹 SHA256 不一致")
if digest(args.native / "initial_states.json") != report["states_sha256"]:
    raise ValueError("原生接触初始状态 SHA256 不一致")
model = build_model(args.robot_model, states["planner_physics"], args.collision_bundle)
data = mujoco.MjData(model)
joint_ids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
object_qpos = int(model.joint("object_free").qposadr[0])
object_geom = model.geom("object").id
body_ids = [model.body(name).id for name in ("gripper", "moving_jaw_so101_v1")]
geometry_ids = [np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0)) for body in body_ids]
if any(not len(values) for values in geometry_ids):
    raise ValueError("两侧夹爪都需要实际碰撞几何")
records = []
with h5py.File(args.native / "trajectory.hdf5") as trajectory:
    for environment in range(report["episodes"]):
        active = trajectory["active"][:, environment]
        native_forces = trajectory["jaw_forces"][:, environment]
        selected = np.flatnonzero(active & (native_forces > .5).all(axis=-1))
        if not len(selected):
            raise ValueError("接触几何检查需要原生双侧实际接触")
        selected = np.unique(np.append(selected[::args.stride], selected[-1]))
        samples = []
        for step in selected:
            data.qpos[joint_ids] = trajectory["joint_position"][step, environment] + JOINT_OFFSETS
            data.qpos[object_qpos:object_qpos + 3] = trajectory["object_position_root"][step, environment]
            data.qpos[object_qpos + 3:object_qpos + 7] = trajectory["object_quaternion_root"][step, environment]
            mujoco.mj_forward(model, data)
            distances = [min(mujoco.mj_geomDistance(model, data, int(geometry), object_geom, .1, None)
                             for geometry in values) for values in geometry_ids]
            if not np.isfinite(distances).all():
                raise RuntimeError("实际夹爪几何距离包含无效数值")
            samples.append({"control_step": int(step), "native_forces_n": native_forces[step].tolist(),
                            "mujoco_signed_surface_distances_m": distances,
                            "jaw_angle_rad": float(trajectory["joint_position"][step, environment, 5])})
        distances = np.asarray([sample["mujoco_signed_surface_distances_m"] for sample in samples])
        records.append({"environment": environment, "sample_count": len(samples),
                        "minimum_distances_m": distances.min(axis=0).tolist(),
                        "maximum_distances_m": distances.max(axis=0).tolist(),
                        "median_distances_m": np.median(distances, axis=0).tolist(), "samples": samples})
result = {"status": "native_contact_geometry_measured", "environments": records, "sampling_stride": args.stride,
          "scope": "MuJoCo_CoACD_surface_distances_at_actual_native_joint_and_object_poses",
          "source_trace_sha256": report["trace_sha256"], "source_states_sha256": report["states_sha256"],
          "robot_model_sha256": digest(args.robot_model), "collision_bundle_sha256": digest(args.collision_bundle / "manifest.json"),
          "source_sha256": digest(Path(__file__)), "contact_equivalence_verified": False}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": result["status"], "environments": [
    {name: value for name, value in record.items() if name != "samples"} for record in records]}, indent=2), flush=True)
