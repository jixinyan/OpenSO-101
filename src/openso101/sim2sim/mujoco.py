import json
import copy
from pathlib import Path

# PyTorch 必须在 CoACD 之前初始化。
from openso101.rl.portable import PortablePolicy

import h5py
import coacd
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from openso101.rl.config import digest

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
JOINT_OFFSETS = np.array([0., -np.pi / 2, np.pi / 2, 0., 0., 0.])


def check_kinematics(model, trace):
    data = mujoco.MjData(model)
    gripper_id = model.body("gripper").id
    qpos_ids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    maximum_position_error = 0.
    maximum_rotation_error = 0.
    positions = trace["joint_position"][:]
    expected_positions = trace["gripper_position_root"][:]
    expected_rotations = trace["gripper_quaternion_root"][:]
    for index in range(len(positions)):
        for env_index in range(positions.shape[1]):
            data.qpos[qpos_ids] = positions[index, env_index] + JOINT_OFFSETS
            mujoco.mj_forward(model, data)
            position_error = np.linalg.norm(data.xpos[gripper_id] - expected_positions[index, env_index])
            predicted = Rotation.from_quat(data.xquat[gripper_id], scalar_first=True)
            expected = Rotation.from_quat(expected_rotations[index, env_index], scalar_first=True)
            rotation_error = (expected.inv() * predicted).magnitude()
            maximum_position_error = max(maximum_position_error, float(position_error))
            maximum_rotation_error = max(maximum_rotation_error, float(rotation_error))
    if maximum_position_error > 0.001 or maximum_rotation_error > 0.001:
        raise ValueError("MuJoCo 关节坐标未通过 Isaac 末端坐标检查")
    return {"poses": int(np.prod(positions.shape[:2])), "maximum_position_error_m": maximum_position_error,
            "maximum_rotation_error_rad": maximum_rotation_error}


def build_model(robot_model, metadata, collision_bundle=None):
    gripper_contact_dimension = metadata.get("gripper_contact_dimension", 3)
    if gripper_contact_dimension not in (3, 4, 6):
        raise ValueError("夹爪接触维度需要使用 MuJoCo 的 3、4 或 6")
    spec = mujoco.MjSpec.from_file(str(robot_model))
    if [joint.name for joint in spec.joints] != list(JOINT_NAMES):
        raise ValueError("需要官方 SO-101 old-calibration MJCF 的六个关节")
    model = spec.compile()
    expected_limits = np.asarray(metadata["physical_joint_limits"]) + JOINT_OFFSETS[:, None]
    if not np.allclose(model.jnt_range, expected_limits, atol=1e-5, rtol=0):
        raise ValueError("MJCF 关节限位与 policy 定义不一致")
    physics_dt = metadata.get("physics_dt", .002)
    if not np.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("policy physics_dt 必须为正数有限数值")
    spec.option.timestep = physics_dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.gravity = [0, 0, -9.81]
    for index, name in enumerate(JOINT_NAMES):
        joint = spec.joint(name)
        joint.damping[:] = 0.
        actuator = spec.actuator(name)
        stiffness = metadata["nominal_stiffness"][index]
        actuator.gainprm[0] = stiffness
        actuator.biasprm[1] = -stiffness
        actuator.biasprm[2] = -metadata["nominal_damping"][index]
        actuator.forcerange = [-metadata["effort_limits"][index], metadata["effort_limits"][index]]
        actuator.forcelimited = True
        actuator.ctrllimited = False
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype = 1
            geom.conaffinity = 6
            if geom.parent.name in ("gripper", "moving_jaw_so101_v1"):
                geom.friction = [1.2, 0.005, 0.0001]
                geom.condim = gripper_contact_dimension
    if collision_bundle is not None:
        bundle = Path(collision_bundle).resolve()
        manifest = json.loads((bundle / "manifest.json").read_text())
        if manifest["schema_version"] != 1 or manifest["generator"] != "CoACD":
            raise ValueError("需要已校验的 CoACD collision bundle")
        replaced = set()
        for geom in list(spec.geoms):
            if not (geom.contype or geom.conaffinity) or geom.meshname not in manifest["meshes"]:
                continue
            name = geom.meshname
            record = manifest["meshes"][name]
            if digest(Path(robot_model).parent / "assets" / f"{name}.stl") != record["source_sha256"]:
                raise ValueError("collision bundle 的原始 STL 校验失败")
            if len(record["parts"]) < 2 or name in replaced:
                raise ValueError("collision bundle 的实体或 convex parts 不完整")
            for index, part in enumerate(record["parts"]):
                path = (bundle / part["file"]).resolve()
                if not path.is_relative_to(bundle) or digest(path) != part["sha256"]:
                    raise ValueError("collision bundle 的 convex part 校验失败")
                part_name = f"collision_{name}_{index:03d}"
                spec.add_mesh(name=part_name, file=str(path))
                geom.parent.add_geom(name=part_name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=part_name,
                                     pos=geom.pos, quat=geom.quat, contype=geom.contype, conaffinity=geom.conaffinity,
                                     friction=geom.friction, condim=geom.condim,
                                     solref=geom.solref, solimp=geom.solimp, group=3)
            geom.contype = 0
            geom.conaffinity = 0
            replaced.add(name)
        if replaced != set(manifest["meshes"]):
            raise ValueError("collision bundle 中的 mesh 未完整应用")
    for extra_index, extra in enumerate(metadata.get("robot_collision_extras", [])):
        vertices = np.asarray(extra["vertices_body"], dtype=float)
        faces = trimesh.geometry.triangulate_quads(extra["polygons"])
        if (extra["body"] != "gripper" or vertices.ndim != 2 or vertices.shape[1] != 3
                or not np.isfinite(vertices).all() or not len(faces)
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("记录的机器人额外 collision mesh 无效")
        coacd.set_log_level("warn")
        parts = coacd.run_coacd(coacd.Mesh(vertices, faces), threshold=.01, preprocess_mode="auto",
                               preprocess_resolution=100, resolution=10000, mcts_iterations=200,
                               mcts_max_depth=4, seed=42)
        if not parts:
            raise RuntimeError("camera mount convex parts 生成失败")
        for part_index, (points, triangles) in enumerate(parts):
            mesh = trimesh.Trimesh(vertices=points, faces=triangles, process=True)
            if not mesh.is_convex or not mesh.is_watertight or not np.isfinite(mesh.vertices).all():
                raise RuntimeError("camera mount convex part 无效")
            name = f"native_camera_mount_{extra_index}_{part_index}"
            spec.add_mesh(name=name, uservert=mesh.vertices.ravel(), userface=mesh.faces.ravel())
            spec.body(extra["body"]).add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                                             contype=1, conaffinity=6, friction=[1.2, .005, .0001],
                                             condim=gripper_contact_dimension, group=3)
    if "table_geometry" in metadata:
        geometry = metadata["table_geometry"]
        position = np.asarray(geometry["position_root"], dtype=float)
        size = np.asarray(geometry["half_size"], dtype=float)
        quaternion = np.asarray(geometry["quaternion_root"], dtype=float)
        if (geometry["type"] != "box" or position.shape != (3,) or size.shape != (3,) or quaternion.shape != (4,)
                or not np.isfinite(position).all() or not np.isfinite(size).all() or (size <= 0).any()
                or not np.isfinite(quaternion).all() or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-6, rtol=0)):
            raise ValueError("记录的桌面 box geometry 无效")
        top = position[2] + np.abs(Rotation.from_quat(quaternion, scalar_first=True).as_matrix()[2]) @ size
        if not np.isclose(top, metadata["table_height_root"], atol=1e-8, rtol=0):
            raise ValueError("桌面高度与记录的 box geometry 不一致")
        spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=size,
                               pos=position, quat=quaternion, contype=4, conaffinity=3, friction=[1, 0.005, 0.0001])
    else:
        spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05],
                               pos=[0, 0, metadata["table_height_root"]], contype=4, conaffinity=3,
                               friction=[1, 0.005, 0.0001])
    cube = spec.worldbody.add_body(name="object", pos=[0.02, -0.3, metadata["table_height_root"] + 0.015])
    cube.add_freejoint(name="object_free")
    cube.add_geom(name="object", type=mujoco.mjtGeom.mjGEOM_BOX, size=np.asarray(metadata["object_size"]) / 2,
                  mass=metadata["object_mass"], contype=2, conaffinity=5, friction=[1, 0.005, 0.0001])
    compiled = spec.compile()
    if collision_bundle is not None or metadata.get("robot_collision_extras"):
        for name in ("body_mass", "body_ipos", "body_inertia", "body_iquat"):
            if not np.allclose(getattr(compiled, name)[:model.nbody], getattr(model, name), atol=1e-12, rtol=0):
                raise RuntimeError("碰撞表示改变了机器人质量、COM 或惯性")
    return compiled


def jaw_forces(model, data):
    object_id = model.geom("object").id
    jaw_ids = [model.body(name).id for name in ("gripper", "moving_jaw_so101_v1")]
    forces = np.zeros((2, 3))
    for index in range(data.ncon):
        contact = data.contact[index]
        if object_id not in (contact.geom1, contact.geom2):
            continue
        robot_geom = contact.geom2 if contact.geom1 == object_id else contact.geom1
        body_id = model.geom_bodyid[robot_geom]
        if body_id in jaw_ids:
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            force_world = contact.frame.reshape(3, 3).T @ wrench[:3]
            forces[jaw_ids.index(body_id)] += force_world if contact.geom1 == object_id else -force_world
    return np.linalg.norm(forces, axis=-1)


def evaluate(args):
    recorded = getattr(args, "recorded_physics", False)
    constrained = getattr(args, "constrained_drive", False)
    if constrained and not recorded:
        raise ValueError("constrained-drive 策略评估需要 recorded-physics")
    if args.episodes <= 0:
        raise ValueError("episodes 必须为正数")
    policy_folder = Path(args.policy).resolve()
    robot_model = Path(args.robot_model).resolve()
    policy = PortablePolicy(policy_folder)
    metadata = policy.metadata
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if recorded:
        from .recorded_physics import COMPONENT_FIELDS, RecordedPhysics

        validation = json.loads((policy_folder / "validation.json").read_text())
        if digest(policy_folder / "isaac_validation.hdf5") != validation["trace_sha256"]:
            raise ValueError("实际物理轨迹 SHA256 与源报告不一致")
    with h5py.File(policy_folder / "isaac_validation.hdf5", "r") as reference:
        kinematics = check_kinematics(mujoco.MjModel.from_xml_path(str(robot_model)), reference)
        if args.episodes > reference["joint_position"].shape[1]:
            raise ValueError("episodes 超过 Isaac 记录的初始场景数量")
        starts = {name: reference[name][0, :args.episodes] for name in (
            "joint_position", "joint_velocity", "object_position_root", "object_quaternion_root", "goal_root",
        )}
        velocity_fields = ("object_linear_velocity_root", "object_angular_velocity_root")
        velocity_available = all(name in reference for name in velocity_fields)
        if any(name in reference for name in velocity_fields) and not velocity_available:
            raise ValueError("物体线速度与角速度需要同时记录")
        if velocity_available:
            starts.update({name: reference[name][0, :args.episodes] for name in velocity_fields})
            if any(starts[name].shape != (args.episodes, 3) or not np.isfinite(starts[name]).all() for name in velocity_fields):
                raise ValueError("记录的物体速度无效")
        if recorded:
            names = {name for component in COMPONENT_FIELDS.values() for name in component}
            names.update(("joint_position", "object_position_root", "object_quaternion_root",
                          "joint_stiffness", "joint_damping", "joint_vel_limits"))
            fields = {name: reference[name][:1] if name == "scene_gravity" else reference[name][:1, :args.episodes] for name in names}
            if any(not np.isfinite(value).all() for value in fields.values()):
                raise ValueError("实际初始环境参数含有无效数值")
    collision_bundle = getattr(args, "collision_bundle", None)
    template = build_model(robot_model, metadata, collision_bundle)
    model = copy.copy(template)
    data = mujoco.MjData(model)
    joint_qpos_ids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    joint_dof_ids = [int(model.joint(name).dofadr[0]) for name in JOINT_NAMES]
    actuator_ids = [model.actuator(name).id for name in JOINT_NAMES]
    object_qpos_id = int(model.joint("object_free").qposadr[0])
    object_dof_id = int(model.joint("object_free").dofadr[0])
    object_id = model.body("object").id
    gripper_id = model.body("gripper").id
    control_dt = metadata["control_dt"]
    substeps = round(control_dt / model.opt.timestep)
    if not np.isclose(substeps * model.opt.timestep, control_dt):
        raise ValueError("MuJoCo physics_dt 无法表示 policy control_dt")
    parameters = metadata["task_parameters"]
    task_height = metadata["task_reference_height_root"] if "task_reference_height_root" in metadata else metadata["table_height_root"]
    records = []
    output.mkdir(parents=True, exist_ok=False)
    with h5py.File(output / "trajectory.hdf5", "w") as trajectory:
        for episode in range(args.episodes):
            model = copy.copy(template)
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            data.qpos[joint_qpos_ids] = starts["joint_position"][episode] + JOINT_OFFSETS
            data.qvel[joint_dof_ids] = starts["joint_velocity"][episode]
            data.qpos[object_qpos_id:object_qpos_id + 3] = starts["object_position_root"][episode]
            data.qpos[object_qpos_id + 3:object_qpos_id + 7] = starts["object_quaternion_root"][episode]
            if velocity_available:
                data.qvel[object_dof_id:object_dof_id+3] = starts["object_linear_velocity_root"][episode]
                data.qvel[object_dof_id+3:object_dof_id+6] = Rotation.from_quat(
                    starts["object_quaternion_root"][episode], scalar_first=True,
                ).inv().apply(starts["object_angular_velocity_root"][episode])
            mujoco.mj_forward(model, data)
            physics = None
            drive = None
            if recorded:
                physics = RecordedPhysics(model, metadata, fields, list(COMPONENT_FIELDS), episode)
                physics.apply(data, 0)
                stiffness = fields["joint_stiffness"][0, episode]
                damping = fields["joint_damping"][0, episode]
                if (stiffness < 0).any() or (damping <= 0).any():
                    raise ValueError("实际 PD 参数需要非负 stiffness 和正数 damping")
                model.actuator_gainprm[actuator_ids, 0] = stiffness
                model.actuator_biasprm[actuator_ids, 1] = -stiffness
                model.actuator_biasprm[actuator_ids, 2] = -damping
            if constrained:
                from .constrained_drive import ConstrainedImplicitDrive

                drive = ConstrainedImplicitDrive(model, actuator_ids, joint_qpos_ids, joint_dof_ids)
                drive.configure(stiffness, damping, fields["joint_vel_limits"][0, episode])
            goal = starts["goal_root"][episode].copy()
            last_action = np.zeros((1, 6), dtype=np.float32)
            stage = 0
            hold_seconds = 0.
            lift_hold_seconds = 0.
            success = False
            progress = {name: False for name in ("reached", "grasped", "held_above_table", "carry_stage", "place_stage")}
            buffers = {name: [] for name in ("joint_position", "joint_velocity", "joint_targets", "object_position_root", "jaw_forces", "raw_action", "stage")}
            physics_buffers = {name: [] for name in ("joint_velocity", "actuator_force", "velocity_command")}
            for step in range(round(metadata["episode_length_s"] / control_dt)):
                qpos = data.qpos[joint_qpos_ids] - JOINT_OFFSETS
                qvel = data.qvel[joint_dof_ids]
                forces = jaw_forces(model, data)
                grasped = bool((forces > metadata["grasp_force_threshold_newtons"]).all())
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, object_id, velocity, 0)
                observation = policy.observation(qpos[None], qvel[None], data.xpos[object_id][None],
                                                 goal[None], [[float(grasped)]], last_action,
                                                 object_velocity=np.concatenate((velocity[3:], velocity[:3]))[None],
                                                 task_state=[[stage / 2., hold_seconds / .5, 1 - step * control_dt / metadata["episode_length_s"]]]
                                                 if metadata["task_id"] == "OpenSO101-PickPlace-v0"
                                                 else [[0., lift_hold_seconds / .25, 1 - step * control_dt / metadata["episode_length_s"]]])
                actions = policy.predict(observation)
                targets = policy.joint_targets(actions, enforce_limits=False, joint_position=qpos[None]).numpy()[0]
                last_action = actions.numpy()
                buffers["joint_position"].append(qpos.copy())
                buffers["joint_velocity"].append(qvel.copy())
                buffers["joint_targets"].append(targets.copy())
                buffers["object_position_root"].append(data.xpos[object_id].copy())
                buffers["jaw_forces"].append(forces.copy())
                buffers["raw_action"].append(last_action[0].copy())
                buffers["stage"].append(stage)
                data.ctrl[actuator_ids] = targets + JOINT_OFFSETS
                for _ in range(substeps):
                    if drive is not None:
                        command = drive.apply(data, targets + JOINT_OFFSETS)
                    mujoco.mj_step(model, data)
                    if drive is not None:
                        drive.verify(data)
                        physics_buffers["joint_velocity"].append(data.qvel[joint_dof_ids].copy())
                        physics_buffers["actuator_force"].append(data.actuator_force[actuator_ids].copy())
                        physics_buffers["velocity_command"].append(command)
                mujoco.mj_forward(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(warning.number for warning in data.warning):
                    raise RuntimeError("MuJoCo 物理运行产生无效状态或警告")
                forces = jaw_forces(model, data)
                grasped = bool((forces > metadata["grasp_force_threshold_newtons"]).all())
                object_position = data.xpos[object_id]
                gripper_rotation = Rotation.from_quat(data.xquat[gripper_id], scalar_first=True)
                ee_position = data.xpos[gripper_id] + gripper_rotation.apply([0.01, 0, -0.09])
                progress["reached"] |= np.linalg.norm(ee_position - object_position) < 0.08
                progress["grasped"] |= grasped
                progress["held_above_table"] |= grasped and object_position[2] > task_height + 0.04
                if metadata["task_id"] == "OpenSO101-Lift-v0":
                    eligible = bool(object_position[2] > task_height + parameters["minimal_height"]
                                    and np.linalg.norm(object_position - goal[:3]) < parameters["goal_radius"])
                    if metadata.get("task_profile", "default") in ("grasp_v2", "grasp_v3", "grasp_v4"):
                        eligible = eligible and grasped
                        lift_hold_seconds = lift_hold_seconds + control_dt if eligible else 0.
                        success = lift_hold_seconds >= parameters["settle_seconds"]
                    else:
                        success = eligible
                else:
                    velocity = np.zeros(6)
                    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, object_id, velocity, 0)
                    released = (stage == 2 and not grasped and data.qpos[joint_qpos_ids[-1]] > parameters["jaw_open_min"]
                                and np.linalg.norm(object_position - parameters["place_goal"]) <= parameters["place_radius"]
                                and np.linalg.norm(velocity[3:]) <= parameters["linear_speed_max"]
                                and np.linalg.norm(velocity[:3]) <= parameters["angular_speed_max"])
                    hold_seconds = hold_seconds + control_dt if released else 0.
                    success = hold_seconds >= parameters["settle_seconds"]
                    if stage < 2 and grasped and np.linalg.norm(object_position - goal) <= parameters["advance_threshold"] + parameters["object_contact_radius"]:
                        stage += 1
                        goal = np.asarray(parameters["place_goal"]).copy()
                        if stage == 1:
                            goal[2] = parameters["carry_height"]
                    progress["carry_stage"] |= stage >= 1
                    progress["place_stage"] |= stage >= 2
                if success or object_position[2] < task_height - 0.05:
                    break
            group = trajectory.create_group(f"episode_{episode:06d}")
            for name, values in buffers.items():
                group.create_dataset(name, data=np.asarray(values))
            records.append({"initial_env_index": episode, "success": success, "steps": step + 1,
                            **{name: bool(value) for name, value in progress.items()}})
            if physics is not None:
                records[-1]["recorded_physics"] = physics.report()
                for name, values in physics.snapshot().items():
                    group.create_dataset(f"model_parameters/{name}", data=values)
            if drive is not None:
                records[-1]["constrained_drive"] = drive.report()
                for name, values in physics_buffers.items():
                    group.create_dataset(f"physics_steps/{name}", data=np.asarray(values))
    report = {
        "status": "mujoco_policy_evaluation_completed", "task": metadata["task_id"],
        "task_profile": metadata.get("task_profile", "default"), "episodes": records,
        "success_rate": sum(record["success"] for record in records) / len(records),
        "kinematics": kinematics, "mujoco_version": mujoco.__version__,
        "policy_sha256": metadata["files"]["policy.pt"], "robot_model_sha256": digest(robot_model),
        "policy_metadata_sha256": digest(policy_folder / "policy.json"),
        "robot_meshes": {path.name: digest(path) for path in sorted((robot_model.parent / "assets").glob("*.stl"))},
        "isaac_trace_sha256": digest(policy_folder / "isaac_validation.hdf5"),
        "trajectory_sha256": digest(output / "trajectory.hdf5"), "control_dt": control_dt,
        "physics_dt": model.opt.timestep, "joint_offsets": JOINT_OFFSETS.tolist(),
        "dynamics": "actual_Isaac_initial_environment_body_gravity_armature_friction_and_PD" if recorded else "nominal_isaac_PD_and_effort_limits_with_upstream_MJCF_inertia_and_frictionloss",
        "parameter_schedule": "fixed_source_initial_environment" if recorded else "nominal",
        "drive_model": "constrained_implicit_PD" if constrained else "position_pd",
        "actual_velocity_limits_verified": constrained and all(record["constrained_drive"]["actual_velocity_limits_verified"] for record in records),
        "constrained_drive_source_sha256": digest(Path(__file__).with_name("constrained_drive.py")) if constrained else None,
        "evaluation_source_sha256": digest(Path(__file__)),
        "recorded_physics_source_sha256": digest(Path(__file__).with_name("recorded_physics.py")) if recorded else None,
        "contact_geometry": "CoACD_gripper_convex_parts" if collision_bundle else "upstream_MJCF_convex_meshes",
        "native_robot_extra_mesh_count": len(metadata.get("robot_collision_extras", [])),
        "native_robot_extra_convex_part_count": sum(model.geom(index).name.startswith("native_camera_mount_")
                                                   for index in range(model.ngeom)),
        "collision_bundle_sha256": digest(Path(collision_bundle) / "manifest.json") if collision_bundle else None,
        "initial_states": "first_frame_of_actual_Isaac_trace",
        "source_object_velocity_available": velocity_available,
        "table_geometry_source": "native_USD_collision_mesh" if "table_geometry" in metadata else "legacy_metadata_plane",
        "task_reference_height_root": task_height,
        "physics_equivalence_verified": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0
