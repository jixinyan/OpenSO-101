import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

from openso101.rl.config import digest
from openso101.sim2sim.mujoco import build_model
from openso101.sim2sim.recorded_physics import RecordedPhysics, COMPONENT_FIELDS


def value_json(value):
    if value is None or isinstance(value, (str, bool, float, int)):
        return value
    return str(value)


def audit(root, output):
    usd = root / "outputs/SO-ARM101-USD.usd"
    stage = Usd.Stage.Open(str(usd))
    model_path = root / "outputs/so-arm100/Simulation/SO101/so101_old_calib.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    urdf_path = model_path.with_suffix(".urdf")
    urdf = ET.parse(urdf_path).getroot()
    prims = []
    for prim in stage.TraverseAll():
        relevant = [name for name in prim.GetAppliedSchemas() if "Physics" in name or "Physx" in name]
        if relevant or prim.IsA(UsdPhysics.Joint):
            prims.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(), "schemas": relevant,
                          "authored_schemas": str(prim.GetMetadata("apiSchemas")),
                          "attributes": {attr.GetName(): value_json(attr.Get()) for attr in prim.GetAttributes()
                                         if attr.GetName().startswith(("physics:", "physx", "drive:"))},
                          "relationships": {rel.GetName(): [str(path) for path in rel.GetTargets()]
                                            for rel in prim.GetRelationships()}})
    collision_meshes = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh) and "/collisions/" in str(prim.GetPath()):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            collision_meshes.append({"path": str(prim.GetPath()), "point_count": len(points),
                                     "local_min": points.min(axis=0).tolist(), "local_max": points.max(axis=0).tolist(),
                                     "enabled": UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                                     "approximation": UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()})
    nominal_bodies = []
    for link in urdf.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        name = link.attrib["name"]
        body = model.body("moving_jaw_so101_v1" if name == "jaw" else name)
        mass = float(inertial.find("mass").attrib["value"])
        com = np.fromstring(inertial.find("origin").attrib["xyz"], sep=" ")
        params = inertial.find("inertia").attrib
        tensor = np.array([[float(params["i" + "".join(sorted(a+b))]) for b in "xyz"] for a in "xyz"])
        rotation = Rotation.from_quat(model.body_iquat[body.id], scalar_first=True).as_matrix()
        mj_tensor = (rotation * model.body_inertia[body.id]) @ rotation.T
        nominal_bodies.append({"name": name, "urdf_mass_kg": mass, "mjcf_mass_kg": float(body.mass[0]),
                               "com_error_m": float(np.linalg.norm(com - model.body_ipos[body.id])),
                               "inertia_relative_error": float(np.linalg.norm(tensor-mj_tensor)/np.linalg.norm(tensor)),
                               "urdf_collisions": len(link.findall("collision")),
                               "mjcf_collision_geoms": int(np.count_nonzero((model.geom_bodyid == body.id) & (model.geom_contype != 0)))})
    sources = {}
    for task in ("lift", "pick_place"):
        folder = root / f"outputs/rl_progress/{task}_body_physics_verified_50"
        metadata = json.loads((folder / "policy.json").read_text())
        with h5py.File(folder / "isaac_validation.hdf5", "r") as trace:
            objects = trace["object_position_root"][:]
            quat = trace["object_quaternion_root"][:]
            rotations = Rotation.from_quat(quat.reshape(-1, 4), scalar_first=True).as_matrix().reshape(*quat.shape[:2], 3, 3)
            vertical_extent = np.abs(rotations[..., 2, :]) @ (np.array(metadata["object_size"])/2)
            gap = objects[..., 2] - vertical_extent - metadata["table_height_root"]
            jaw_forces = trace["jaw_forces"][:]
            relative = objects - trace["gripper_position_root"][:]
            grip_rot = Rotation.from_quat(trace["gripper_quaternion_root"][:].reshape(-1, 4), scalar_first=True)
            local_objects = grip_rot.inv().apply(relative.reshape(-1, 3)).reshape(relative.shape)
            distance = np.linalg.norm(local_objects - [0.01, 0, -0.09], axis=-1)
            sources[task] = {"trace_sha256": digest(folder / "isaac_validation.hdf5"),
                             "table_height_root_m": metadata["table_height_root"],
                             "object_bottom_minus_table_m": {"minimum": float(gap.min()), "maximum": float(gap.max()),
                                                               "last_mean": float(gap[-1].mean())},
                             "maximum_object_tilt_rad": float(np.arccos(np.clip(rotations[..., 2, 2], -1, 1)).max()),
                             "minimum_grasp_center_distance_m": float(distance.min()),
                             "minimum_jaw_position_rad": float(trace["joint_position"][:, :, 5].min()),
                             "jaw_target_range_rad": [float(trace["joint_targets"][:, :, 5].min()), float(trace["joint_targets"][:, :, 5].max())],
                             "bilateral_contact_samples": int(((jaw_forces > 0.5).all(axis=-1)).sum()),
                             "material_shape": list(trace["robot_material_properties"].shape),
                             "material_minimum": trace["robot_material_properties"][:].min(axis=(0, 1, 2)).tolist(),
                             "material_maximum": trace["robot_material_properties"][:].max(axis=(0, 1, 2)).tolist()}
    folder = root / "outputs/rl_progress/lift_body_physics_verified_50"
    metadata = json.loads((folder / "policy.json").read_text())
    with h5py.File(folder / "isaac_validation.hdf5", "r") as trace:
        fields = {name: trace[name][:1] for name in COMPONENT_FIELDS["bodies"]}
        fields.update({name: trace[name][:1] for name in ("joint_position", "object_position_root", "object_quaternion_root")})
    geometry_model = build_model(model_path, metadata)
    converter = RecordedPhysics(geometry_model, metadata, fields, ["bodies"], 0)
    cache = UsdGeom.XformCache()
    geometry = []
    for index, name in enumerate(metadata["physics_recording"]["robot_body_names"]):
        link = stage.GetPrimAtPath(f"/so101_new_calib/{name}")
        inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(link)))
        native_points = []
        native_parts = []
        for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh) and "/collisions/" in str(prim.GetPath()):
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                homogeneous = np.column_stack((points, np.ones(len(points))))
                local = (homogeneous @ np.asarray(cache.GetLocalToWorldTransform(prim)) @ inverse)[:, :3]
                converted = local @ converter.frame_rotation[index].T + converter.frame_translation[index]
                native_points.append(converted)
                native_parts.append((str(prim.GetPath()), converted))
        body_id = converter.body_ids[index]
        mj_points = []
        reference = converter.reference
        for geom_id in np.flatnonzero((geometry_model.geom_bodyid == body_id) & (geometry_model.geom_contype == 1)):
            mesh_id = geometry_model.geom_dataid[geom_id]
            start = geometry_model.mesh_vertadr[mesh_id]
            points = geometry_model.mesh_vert[start:start+geometry_model.mesh_vertnum[mesh_id]]
            world = points @ reference.geom_xmat[geom_id].reshape(3, 3).T + reference.geom_xpos[geom_id]
            mj_points.append((world-reference.xpos[body_id]) @ reference.xmat[body_id].reshape(3, 3))
        if not mj_points:
            geometry.append({"name": name, "official_mjcf_collision_present": False})
            continue
        native_points = np.concatenate(native_points)
        mj_points = np.concatenate(mj_points)
        distances = np.concatenate((cKDTree(native_points).query(mj_points)[0], cKDTree(mj_points).query(native_points)[0]))
        geometry.append({"name": name, "usd_points": len(native_points), "mjcf_points": len(mj_points),
                         "bidirectional_vertex_max_distance_m": float(distances.max()),
                         "bidirectional_vertex_95_percentile_distance_m": float(np.quantile(distances, .95)),
                         "usd_parts": [{"path": path, "maximum_vertex_distance_to_official_mjcf_m": float(cKDTree(mj_points).query(points)[0].max())}
                                       for path, points in native_parts]})
    report = {"usd_sha256": digest(usd), "mjcf_sha256": digest(model_path), "urdf_sha256": digest(urdf_path),
              "usd_meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage), "usd_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
              "usd_physics_prims": prims, "usd_collision_meshes": collision_meshes,
              "official_urdf_mjcf_bodies": nominal_bodies, "collision_source_mesh_comparison": geometry,
              "official_mjcf_effort_limits_nm": model.actuator_forcerange.tolist(),
              "official_mjcf_armature": model.dof_armature.tolist(),
              "official_mjcf_frictionloss": model.dof_frictionloss.tolist(), "source_trajectories": sources}
    output.mkdir(parents=True, exist_ok=False)
    (output / "model_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"physics_prims": len(prims), "collision_meshes": len(collision_meshes), "bodies": nominal_bodies,
                      "source_trajectories": sources}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(Path(__file__).resolve().parents[1], args.output)
