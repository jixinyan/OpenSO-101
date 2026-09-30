import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from openso101.rl.config import digest
from openso101.sim2sim.mujoco import JOINT_NAMES, JOINT_OFFSETS, build_model


CONDITIONS = ("baseline", "bodies", "gravity", "armature", "friction", "combined")
TASKS = ("lift", "pick_place")


def check(root, run_id, output):
    if output.exists():
        raise FileExistsError(output)
    summary = {"paired_inputs_verified": True, "parameter_readback_verified": True,
               "task_success_verified": False, "physics_equivalence_verified": False, "tasks": {}}
    for task in TASKS:
        policy = root / f"{task}_body_physics_verified_50"
        metadata = json.loads((policy / "policy.json").read_text())
        model = build_model(Path("outputs/so-arm100/Simulation/SO101/so101_old_calib.xml").resolve(), metadata)
        body_ids = [model.body("moving_jaw_so101_v1" if name == "jaw" else name).id
                    for name in metadata["physics_recording"]["robot_body_names"]]
        object_id = model.body("object").id
        qpos_ids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
        source_hash = digest(policy / "isaac_validation.hdf5")
        task_results = {}
        baseline_folder = root / f"{task}_physics_components_{run_id}_baseline"
        baseline_report = json.loads((baseline_folder / "report.json").read_text())
        with h5py.File(baseline_folder / "comparison.hdf5", "r") as baseline:
            for condition in CONDITIONS:
                folder = root / f"{task}_physics_components_{run_id}_{condition}"
                report = json.loads((folder / "report.json").read_text())
                assert digest(folder / "comparison.hdf5") == report["trajectory_sha256"]
                assert report["isaac_trace_sha256"] == source_hash
                for name in ("policy_sha256", "policy_metadata_sha256", "robot_model_sha256", "robot_meshes",
                             "control_dt", "physics_dt", "comparison_source_sha256", "model_builder_sha256",
                             "drive_model", "pd_source", "velocity_servo_source_sha256"):
                    assert report[name] == baseline_report[name], name
                maximum_com_error = maximum_inertia_error = maximum_force_error = 0.0
                joint_rmse = []
                sample_count = 0
                with h5py.File(folder / "comparison.hdf5", "r") as trajectory:
                    assert len(trajectory) == 4
                    for index, record in enumerate(report["environments"]):
                        group = trajectory[f"environment_{index:06d}"]
                        base = baseline[group.name]
                        steps = record["steps"]
                        sample_count += steps
                        for name in ("joint_targets", "isaac/joint_position", "isaac/joint_velocity",
                                     "isaac/object_position_root", "isaac/jaw_forces", "isaac/joint_stiffness",
                                     "isaac/joint_damping", "isaac/joint_vel_limits"):
                            assert np.array_equal(group[name][:], base[name][:]), name
                        assert record["initial_joint_position_error_rad"] == 0
                        assert record["initial_joint_velocity_error_rad_s"] == 0
                        error = group["mujoco/joint_position"][:] - group["isaac/joint_position"][:]
                        joint_rmse.extend(np.mean(error ** 2, axis=1).tolist())
                        substeps = round(report["control_dt"] / report["physics_dt"])
                        command = group["physics_steps/velocity_command"][:]
                        before = group["physics_steps/joint_velocity_before"][:]
                        damping = np.repeat(group["isaac/joint_damping"][:], substeps, axis=0)
                        expected_force = np.clip(damping * (command - before),
                                                 -np.asarray(metadata["effort_limits"]), metadata["effort_limits"])
                        force_error = float(np.max(np.abs(expected_force - group["physics_steps/actuator_force"][:])))
                        assert force_error < 1e-10
                        maximum_force_error = max(maximum_force_error, force_error)
                        parameters = group["model_parameters"]
                        components = report["physics_components"]
                        parameter_components = {"body_mass": "bodies", "body_ipos": "bodies", "body_inertia": "bodies",
                                                "body_iquat": "bodies", "gravity": "gravity",
                                                "joint_armature": "armature", "joint_frictionloss": "friction"}
                        for name, component in parameter_components.items():
                            if component not in components:
                                assert np.array_equal(parameters[name][:], base[f"model_parameters/{name}"][:]), name
                        assert record["recorded_physics"]["verified_control_steps"] == steps
                        assert record["recorded_physics"]["running_state_preserved_verified"]
                        if "gravity" in components:
                            assert np.array_equal(parameters["gravity"][:], group["isaac/scene_gravity"][:])
                        if "armature" in components:
                            assert np.array_equal(parameters["joint_armature"][:], group["isaac/joint_armature"][:])
                        if "friction" in components:
                            assert np.count_nonzero(parameters["joint_frictionloss"][:]) == 0
                            assert np.count_nonzero(group["isaac/joint_friction_coeff"][:]) == 0
                        if "bodies" in components:
                            assert np.array_equal(parameters["body_mass"][:, body_ids], group["isaac/robot_body_mass"][:])
                            assert np.array_equal(parameters["body_mass"][:, object_id], group["isaac/object_body_mass"][:, 0])
                            source_matrices = group["isaac/robot_body_inertia"][:].reshape(steps, -1, 3, 3).swapaxes(-1, -2)
                            source_rotations = Rotation.from_quat(group["isaac/robot_body_quaternion_root"][:].reshape(-1, 4),
                                                                 scalar_first=True).as_matrix().reshape(steps, -1, 3, 3)
                            data = mujoco.MjData(model)
                            for step in range(steps):
                                for name in ("body_mass", "body_ipos", "body_inertia", "body_iquat"):
                                    getattr(model, name)[:] = parameters[name][step]
                                data.qpos[qpos_ids] = group["isaac/joint_position"][step] + JOINT_OFFSETS
                                mujoco.mj_forward(model, data)
                                rotation = source_rotations[step]
                                source_com = group["isaac/robot_body_position_root"][step] + np.einsum(
                                    "bij,bj->bi", rotation, group["isaac/robot_body_com"][step, :, :3])
                                com_error = float(np.max(np.linalg.norm(source_com - data.xipos[body_ids], axis=-1)))
                                source_inertia = rotation @ source_matrices[step] @ rotation.swapaxes(-1, -2)
                                actual_rotation = data.ximat[body_ids].reshape(-1, 3, 3)
                                actual_inertia = (actual_rotation * model.body_inertia[body_ids, None, :]) @ actual_rotation.swapaxes(-1, -2)
                                inertia_error = float(np.max(np.linalg.norm(source_inertia - actual_inertia, axis=(-1, -2)) /
                                                             np.linalg.norm(source_inertia, axis=(-1, -2))))
                                assert com_error < 1e-5 and inertia_error < 1e-4
                                maximum_com_error = max(maximum_com_error, com_error)
                                maximum_inertia_error = max(maximum_inertia_error, inertia_error)
                            object_inertia = group["isaac/object_body_inertia"][:].reshape(steps, 3, 3).swapaxes(-1, -2)
                            actual_moments = np.sort(parameters["body_inertia"][:, object_id], axis=-1)
                            assert np.allclose(actual_moments, np.linalg.eigvalsh(object_inertia), atol=1e-12, rtol=1e-6)
                            assert np.array_equal(parameters["body_ipos"][:, object_id], group["isaac/object_body_com"][:, :3])
                task_results[condition] = {
                    "environments": 4, "control_samples": sample_count,
                    "joint_position_rmse_rad": float(np.sqrt(np.mean(joint_rmse))),
                    "joint_position_max_error_rad": max(max(e["joint_position_max_error_rad"]) for e in report["environments"]),
                    "physics_max_joint_speed_rad_s": max(max(e["velocity_servo"]["max_actual_speed_rad_s"]) for e in report["environments"]),
                    "object_position_max_error_m": max(e["object_position_max_error_m"] for e in report["environments"]),
                    "maximum_com_error_m": maximum_com_error if "bodies" in report["physics_components"] else None,
                    "maximum_inertia_relative_error": maximum_inertia_error if "bodies" in report["physics_components"] else None,
                    "maximum_force_equation_error_nm": maximum_force_error,
                    "source_trace_sha256": source_hash, "trajectory_sha256": report["trajectory_sha256"],
                }
                print(task, condition, task_results[condition], flush=True)
        summary["tasks"][task] = task_results
    summary["checker_sha256"] = digest(Path(__file__))
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/rl_progress"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.root, args.run_id, args.output)
