import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from openso101.rl.config import digest
from openso101.sim2sim.mujoco import JOINT_NAMES, JOINT_OFFSETS, build_model


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=Path, required=True)
parser.add_argument("--policy", type=Path, required=True)
parser.add_argument("--robot-model", type=Path, required=True)
parser.add_argument("--collision-bundle", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
report = json.loads((args.task / "report.json").read_text())
states = json.loads((args.task / "initial_states.json").read_text())
plan = json.loads((args.task / "plan.json").read_text())
metadata = json.loads((args.policy / "policy.json").read_text())
if digest(args.task / "trajectory.hdf5") != report["trace_sha256"]:
    raise ValueError("原生轨迹 SHA256 检查失败")
model = build_model(args.robot_model, metadata, args.collision_bundle)
data = mujoco.MjData(model)
joint_qpos = [model.joint(name).qposadr[0] for name in JOINT_NAMES]
joint_dof = [model.joint(name).dofadr[0] for name in JOINT_NAMES]
object_qpos = model.joint("object_free").qposadr[0]


def inspect(position, object_position):
    mujoco.mj_resetData(model, data)
    data.qpos[joint_qpos] = np.asarray(position) + JOINT_OFFSETS
    data.qpos[object_qpos:object_qpos + 3] = object_position
    data.qpos[object_qpos + 3:object_qpos + 7] = [1., 0., 0., 0.]
    mujoco.mj_forward(model, data)
    contacts = []
    for contact in data.contact:
        if contact.dist >= -.0001:
            continue
        bodies = [model.body(model.geom_bodyid[index]).name for index in (contact.geom1, contact.geom2)]
        geometries = [model.geom(index).name for index in (contact.geom1, contact.geom2)]
        if "table" in geometries and set(bodies) <= {"world", "base", "object"}:
            continue
        contacts.append({"bodies": bodies, "geometries": geometries, "penetration_m": float(-contact.dist)})
    return {"holding_effort_nm": data.qfrc_bias[joint_dof].tolist(), "contacts": contacts}


records = []
with h5py.File(args.task / "trajectory.hdf5", "r") as trace:
    for index, environment in enumerate(plan["environments"]):
        valid = np.flatnonzero(trace["active"][:, index])
        record = {"environment": index, "initial": inspect(states["environments"][index]["joint_position"],
                                                             states["environments"][index]["object_position_root"]),
                  "final": inspect(trace["joint_position"][valid[-1], index], trace["object_position_root"][valid[-1], index]),
                  "planned_waypoints": []}
        for target in environment["targets"]:
            qpos = target["joint_position"] + [.8 if target["phase"] != "lift" else 0.]
            record["planned_waypoints"].append({"phase": target["phase"], **inspect(
                qpos, states["environments"][index]["object_position_root"])})
        records.append(record)
result = {"status": "geometry_contacts_inspected", "environments": records,
          "native_trace_sha256": report["trace_sha256"], "robot_model_sha256": digest(args.robot_model),
          "collision_bundle_sha256": digest(args.collision_bundle / "manifest.json"),
          "table_geometry": metadata["table_geometry"], "source_sha256": digest(Path(__file__)),
          "native_contact_validation": False}
with args.output.open("x") as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
