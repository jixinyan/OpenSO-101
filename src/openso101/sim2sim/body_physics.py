import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from openso101.rl.config import digest
from .mujoco import JOINT_NAMES, JOINT_OFFSETS, build_model, check_kinematics


def compare_body_physics(args):
    folder = Path(args.policy).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    metadata = json.loads((folder / "policy.json").read_text())
    validation = json.loads((folder / "validation.json").read_text())
    trace_path = folder / "isaac_validation.hdf5"
    if validation["trace_sha256"] != digest(trace_path):
        raise ValueError("实际物理轨迹 SHA256 与源报告不一致")
    recording = metadata["physics_recording"]
    if recording["inertia_frame"] != "body_prim_at_center_of_mass" or recording["inertia_matrix_order"] != "column_major":
        raise ValueError("需要原生 body COM 处的 column-major 惯性记录")
    body_names = recording["robot_body_names"]
    robot_model = Path(args.robot_model).resolve()
    model = build_model(robot_model, metadata)
    data = mujoco.MjData(model)
    body_mapping = {name: ("moving_jaw_so101_v1" if name == "jaw" else name) for name in body_names}
    body_ids = [model.body(body_mapping[name]).id for name in body_names]
    qpos_ids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    dof_ids = [int(model.joint(name).dofadr[0]) for name in JOINT_NAMES]
    with h5py.File(trace_path) as trace:
        kinematics = check_kinematics(model, trace)
        fields = {name: trace[name][:] for name in (
            "joint_position", "robot_body_mass", "robot_body_inertia", "robot_body_com",
            "robot_body_position_root", "robot_body_quaternion_root", "robot_material_properties",
            "joint_armature", "joint_friction_coeff", "object_body_mass", "object_body_inertia",
            "object_material_properties", "scene_gravity",
        )}
    if any(not np.isfinite(value).all() for value in fields.values()):
        raise ValueError("实际物理记录包含非有限数值")
    steps, environments, joints = fields["joint_position"].shape
    bodies = len(body_names)
    expected_shapes = {"robot_body_mass": (steps, environments, bodies),
                       "robot_body_inertia": (steps, environments, bodies, 9),
                       "robot_body_com": (steps, environments, bodies, 7),
                       "robot_body_position_root": (steps, environments, bodies, 3),
                       "robot_body_quaternion_root": (steps, environments, bodies, 4),
                       "joint_armature": (steps, environments, 6),
                       "joint_friction_coeff": (steps, environments, 6), "scene_gravity": (steps, 3)}
    if joints != 6 or any(fields[name].shape != shape for name, shape in expected_shapes.items()):
        raise ValueError("实际物理字段形状与实体列表不一致")
    native_inertia = fields["robot_body_inertia"].reshape(steps, environments, bodies, 3, 3).swapaxes(-1, -2)
    if not np.allclose(native_inertia, native_inertia.swapaxes(-1, -2), atol=1e-10, rtol=1e-5):
        raise ValueError("实际惯性矩阵需要对称")
    moments = np.linalg.eigvalsh(native_inertia)
    if (fields["robot_body_mass"] <= 0).any() or (moments <= 0).any():
        raise ValueError("实际质量与主惯性矩需要正数")
    com_errors = np.empty((steps, environments, bodies))
    position_errors = np.empty_like(com_errors)
    rotation_errors = np.empty_like(com_errors)
    inertia_relative_errors = np.empty_like(com_errors)
    for step in range(steps):
        for environment in range(environments):
            data.qpos[qpos_ids] = fields["joint_position"][step, environment] + JOINT_OFFSETS
            mujoco.mj_forward(model, data)
            native_rotations = Rotation.from_quat(fields["robot_body_quaternion_root"][step, environment], scalar_first=True)
            native_matrices = native_rotations.as_matrix()
            local_com = fields["robot_body_com"][step, environment, :, :3]
            native_com = fields["robot_body_position_root"][step, environment] + native_rotations.apply(local_com)
            com_errors[step, environment] = np.linalg.norm(native_com - data.xipos[body_ids], axis=-1)
            position_errors[step, environment] = np.linalg.norm(fields["robot_body_position_root"][step, environment] - data.xpos[body_ids], axis=-1)
            rotation_errors[step, environment] = (native_rotations.inv() * Rotation.from_quat(data.xquat[body_ids], scalar_first=True)).magnitude()
            native_root_inertia = native_matrices @ native_inertia[step, environment] @ native_matrices.swapaxes(-1, -2)
            inertia_rotations = data.ximat[body_ids].reshape(bodies, 3, 3)
            mujoco_root_inertia = (inertia_rotations * model.body_inertia[body_ids, None, :]) @ inertia_rotations.swapaxes(-1, -2)
            inertia_relative_errors[step, environment] = np.linalg.norm(native_root_inertia - mujoco_root_inertia, axis=(-1, -2)) / np.linalg.norm(mujoco_root_inertia, axis=(-1, -2))
    records = []
    for index, name in enumerate(body_names):
        mass = fields["robot_body_mass"][..., index]
        records.append({
            "body_name": name, "mujoco_body_name": body_mapping[name], "mujoco_body_id": body_ids[index],
            "isaac_mass_range_kg": [float(mass.min()), float(mass.max())],
            "mujoco_mass_kg": float(model.body_mass[body_ids[index]]),
            "isaac_to_mujoco_mass_ratio_range": [float(mass.min() / model.body_mass[body_ids[index]]), float(mass.max() / model.body_mass[body_ids[index]])],
            "isaac_principal_moment_range_kg_m2": [moments[..., index, :].min(axis=(0, 1)).tolist(), moments[..., index, :].max(axis=(0, 1)).tolist()],
            "mujoco_principal_moments_kg_m2": sorted(model.body_inertia[body_ids[index]].tolist()),
            "maximum_body_position_difference_m": float(position_errors[..., index].max()),
            "maximum_body_rotation_difference_rad": float(rotation_errors[..., index].max()),
            "maximum_com_position_difference_m": float(com_errors[..., index].max()),
            "maximum_root_frame_inertia_relative_difference": float(inertia_relative_errors[..., index].max()),
        })
    native_materials = {name: {"minimum": fields[f"{name}_material_properties"].min(axis=tuple(range(fields[f"{name}_material_properties"].ndim - 1))).tolist(),
                               "maximum": fields[f"{name}_material_properties"].max(axis=tuple(range(fields[f"{name}_material_properties"].ndim - 1))).tolist()}
                        for name in ("robot", "object")}
    report = {
        "status": "actual_body_physics_comparison_completed", "task": metadata["task_id"],
        "state_samples": steps * environments, "steps": steps, "environments": environments,
        "bodies": records, "kinematics": kinematics,
        "joint_parameters": {
            "isaac_armature_range": [fields["joint_armature"].min(axis=(0, 1)).tolist(), fields["joint_armature"].max(axis=(0, 1)).tolist()],
            "mujoco_armature": model.dof_armature[dof_ids].tolist(),
            "isaac_friction_coefficient_range": [fields["joint_friction_coeff"].min(axis=(0, 1)).tolist(), fields["joint_friction_coeff"].max(axis=(0, 1)).tolist()],
            "mujoco_frictionloss_nm": model.dof_frictionloss[dof_ids].tolist(),
        },
        "gravity": {"isaac_min_m_s2": fields["scene_gravity"].min(axis=0).tolist(), "isaac_max_m_s2": fields["scene_gravity"].max(axis=0).tolist(), "mujoco_m_s2": model.opt.gravity.tolist()},
        "object_mass": {"isaac_range_kg": [float(fields["object_body_mass"].min()), float(fields["object_body_mass"].max())], "mujoco_kg": float(model.body_mass[model.body("object").id])},
        "materials": {"isaac_components": recording["material_components"], "isaac_native_ranges": native_materials,
                      "mujoco_components": ["sliding_friction", "torsional_friction", "rolling_friction"],
                      "mujoco_geoms": [{"name": model.geom(index).name, "body": model.body(int(model.geom_bodyid[index])).name, "friction": model.geom_friction[index].tolist()}
                                        for index in range(model.ngeom) if model.geom_contype[index] or model.geom_conaffinity[index]],
                      "contact_equivalence_verified": False},
        "isaac_trace_sha256": digest(trace_path), "policy_metadata_sha256": digest(folder / "policy.json"),
        "robot_model_sha256": digest(robot_model), "comparison_source_sha256": digest(Path(__file__)),
        "model_builder_sha256": digest(Path(__file__).with_name("mujoco.py")),
        "mujoco_version": mujoco.__version__, "physics_equivalence_verified": False, "task_success_verified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as target:
        target.write(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))
    return 0
