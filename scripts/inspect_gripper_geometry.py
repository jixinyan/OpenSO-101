import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from openso101.sim2sim.mujoco import JOINT_NAMES, JOINT_OFFSETS, build_model


parser = argparse.ArgumentParser()
parser.add_argument("--plan", type=Path, required=True)
parser.add_argument("--collision-bundle", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--trajectory", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
metadata = json.loads((root / "outputs/rl_progress/lift_scene_geometry_oriented_verified/policy.json").read_text())
plan = json.loads(args.plan.read_text())
records = []
for bundle in (None, args.collision_bundle):
    model = build_model(root / "outputs/so-arm100/Simulation/SO101/so101_old_calib.xml", metadata, bundle)
    data = mujoco.MjData(model)
    ids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    close_index = max(index for index, phase in enumerate(plan["phases"]) if phase == "close")
    data.qpos[ids] = np.asarray(plan["joint_targets"])[close_index] + JOINT_OFFSETS
    oq = int(model.joint("object_free").qposadr[0])
    data.qpos[oq:oq+3] = plan["object_position_root"]
    data.qpos[oq+3:oq+7] = plan["object_quaternion_root"]
    if args.trajectory:
        import h5py

        with h5py.File(args.trajectory, "r") as trace:
            data.qpos[ids] = trace["joint_position"][close_index] + JOINT_OFFSETS
            data.qpos[oq:oq+3] = trace["object_position_root"][close_index]
    mujoco.mj_forward(model, data)
    gid = model.body("gripper").id
    rotation = data.xmat[gid].reshape(3, 3)
    object_id = model.geom("object").id
    contacts = []
    bounds = {}
    for body in ("gripper", "moving_jaw_so101_v1"):
        body_id = model.body(body).id
        active = np.flatnonzero((model.geom_bodyid == body_id) & (model.geom_contype > 0))
        vertices = []
        for index in active:
            mesh_id = model.geom_dataid[index]
            offset, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            world = model.mesh_vert[offset:offset+count] @ data.geom_xmat[index].reshape(3, 3).T + data.geom_xpos[index]
            vertices.append((world - data.xpos[gid]) @ rotation)
            fromto = np.zeros(6)
            distance = mujoco.mj_geomDistance(model, data, int(index), object_id, .5, fromto)
            contacts.append({"body": body, "geom": int(index), "name": model.geom(int(index)).name,
                             "distance_m": distance, "nearest_points_root": fromto.tolist()})
        vertices = np.concatenate(vertices)
        bounds[body] = {"minimum_gripper_local": vertices.min(axis=0).tolist(), "maximum_gripper_local": vertices.max(axis=0).tolist()}
    minimum = {body: min((record for record in contacts if record["body"] == body), key=lambda record: record["distance_m"])
               for body in ("gripper", "moving_jaw_so101_v1")}
    records.append({"collision_bundle": str(bundle), "bounds": bounds, "minimum_distance": minimum,
                    "object_contacts": [{"geom1": int(data.contact[index].geom1), "geom2": int(data.contact[index].geom2),
                                         "distance": float(data.contact[index].dist), "efc_address": int(data.contact[index].efc_address),
                                         "solref": data.contact[index].solref.tolist(), "solimp": data.contact[index].solimp.tolist()} for index in range(data.ncon)
                                        if object_id in (data.contact[index].geom1, data.contact[index].geom2)],
                    "contacts": data.ncon, "geom_masks": {body: model.geom_contype[model.geom_bodyid == model.body(body).id].tolist() for body in ("gripper", "moving_jaw_so101_v1")},
                    "object_position_gripper_local": ((data.qpos[oq:oq+3]-data.xpos[gid]) @ rotation).tolist(),
                    "gripper_rotation_root": rotation.tolist()})
args.output.write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records), flush=True)
