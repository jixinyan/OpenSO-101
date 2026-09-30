import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from openso101.rl.config import digest
from openso101.sim2sim.mujoco import build_model


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
root = Path(__file__).resolve().parents[1]
folder = root / "outputs/rl_progress"
audit_path = folder / "model_debug_parts/model_audit.json"
audit = json.loads(audit_path.read_text())
native_path = folder / "native_scene_final_debug_report.json"
native = json.loads(native_path.read_text())
visuals_path = folder / "model_debug_visuals_final/visualization_report.json"
visuals = json.loads(visuals_path.read_text())
assert native["num_envs"] == 4
assert native["root_cloning_position_error_m"] <= 1e-5
assert native["fixed_root_drift_m"] <= 1e-6
assert native["fixed_root_quaternion_drift"] <= 1e-6
assert abs(native["settled_cube_center_world_z_m"]-.012) < 1e-6
assert audit["usd_meters_per_unit"] == 1 and audit["usd_up_axis"] == "Z"
assert all(body["urdf_mass_kg"] == body["mjcf_mass_kg"] and body["com_error_m"] < 1e-10
           and body["inertia_relative_error"] < 1e-10 for body in audit["official_urdf_mjcf_bodies"])
for row in audit["collision_source_mesh_comparison"]:
    for part in row.get("usd_parts", []):
        if "camera_mount" not in part["path"]:
            assert part["maximum_vertex_distance_to_official_mjcf_m"] < 3e-6
records = []
for task in ("lift", "pick_place"):
    policy_path = folder / f"{task}_measured_table_policy.json"
    metadata = json.loads(policy_path.read_text())
    validation_path = folder / f"{task}_scene_geometry_validation.json"
    validation = json.loads(validation_path.read_text())
    assert metadata["table_geometry"] == native["table_geometry"]
    assert validation["validation_steps"] == 500 and validation["num_envs"] == 4
    assert max(validation["maximum_errors"].values()) < 1e-5
    assert validation["trace_sha256"] == digest(folder / f"{task}_body_physics_verified_50/isaac_validation.hdf5")
    assert metadata["robot_usd_sha256"] == audit["usd_sha256"]
    assert abs(metadata["task_reference_height_root"]-metadata["table_height_root"]-.003) < 1e-10
    model = build_model(root / "outputs/so-arm100/Simulation/SO101/so101_old_calib.xml", metadata)
    table = model.geom("table")
    assert model.geom_type[table.id] == mujoco.mjtGeom.mjGEOM_BOX
    assert np.allclose(model.geom_pos[table.id], metadata["table_geometry"]["position_root"], atol=1e-10, rtol=0)
    assert np.allclose(model.geom_size[table.id], metadata["table_geometry"]["half_size"], atol=1e-10, rtol=0)
    actual_rotation = Rotation.from_quat(model.geom_quat[table.id], scalar_first=True).as_matrix()
    source_rotation = Rotation.from_quat(metadata["table_geometry"]["quaternion_root"], scalar_first=True).as_matrix()
    assert np.allclose(actual_rotation, source_rotation, atol=1e-10, rtol=0)
    comparison_path = folder / f"{task}_scene_geometry_final_compared_report.json"
    comparison = json.loads(comparison_path.read_text())
    assert comparison["isaac_trace_sha256"] == validation["trace_sha256"]
    assert comparison["policy_metadata_sha256"] == digest(policy_path)
    assert comparison["table_geometry_source"] == "native_USD_collision_mesh"
    assert comparison["table_geometry"] == metadata["table_geometry"]
    geometry = comparison["table_geometry_mujoco"]
    assert geometry["position"] == model.geom_pos[table.id].tolist()
    assert geometry["half_size"] == model.geom_size[table.id].tolist()
    assert np.allclose(geometry["quaternion"], model.geom_quat[table.id], atol=1e-10, rtol=0)
    assert comparison["source_object_velocity_available"]
    evaluation_path = folder / f"{task}_scene_geometry_final_evaluated_report.json"
    evaluation = json.loads(evaluation_path.read_text())
    assert evaluation["policy_metadata_sha256"] == digest(policy_path)
    assert evaluation["source_object_velocity_available"]
    assert evaluation["table_geometry_source"] == "native_USD_collision_mesh"
    assert evaluation["task_reference_height_root"] == metadata["task_reference_height_root"]
    assert len(evaluation["episodes"]) == 4
    records.append({"task": task, "validation_sha256": digest(validation_path),
                    "comparison_sha256": digest(comparison_path), "evaluation_sha256": digest(evaluation_path),
                    "maximum_joint_error_rad": max(max(row["joint_position_max_error_rad"]) for row in comparison["environments"]),
                    "maximum_object_error_m": max(row["object_position_max_error_m"] for row in comparison["environments"]),
                    "closed_loop_success_rate": evaluation["success_rate"],
                    "policy_metadata_sha256": digest(policy_path)})
for name, expected in visuals["figures"].items():
    path = folder / "model_debug_visuals_final" / name
    assert digest(path) == expected
    with Image.open(path) as frame:
        frame.verify()
report = {"status": "so101_debug_checks_passed", "native_scene_report_sha256": digest(native_path),
          "model_audit_sha256": digest(audit_path), "visualization_report_sha256": digest(visuals_path),
          "tasks": records, "source_code_sha256": digest(Path(__file__)),
          "physics_equivalence_verified": False, "task_success_verified": False}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
