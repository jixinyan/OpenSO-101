import argparse
import json
from pathlib import Path
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--planner-python", type=Path, required=True)
parser.add_argument("--robot-model", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--task-profile", choices=("grasp_v3", "grasp_v4"), default="grasp_v3")
parser.add_argument("--task", choices=("OpenSO101-Lift-v0", "OpenSO101-PickPlace-v0"), default="OpenSO101-Lift-v0")
parser.add_argument("--with-cameras", action="store_true")
parser.add_argument("--camera-resolution", type=int, default=256)
args = parser.parse_args()
if args.output.exists() or args.num_envs <= 0:
    raise ValueError("任务检查需要新的输出目录和有效环境数量")
args.output.mkdir(parents=True, exist_ok=False)
pick_place = args.task == "OpenSO101-PickPlace-v0"
if pick_place and args.task_profile != "grasp_v4":
    raise ValueError("完整 PickPlace 检查需要 grasp_v4 控制配置")
args.environment_mode = "nominal"
args.visual_dr = False

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=args.with_cameras).app
env = None
try:
    import h5py
    import numpy as np
    import torch
    from isaaclab.managers import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
    from isaaclab.managers.recorder_manager import DatasetExportMode
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

    from openso101.rl.config import digest
    from openso101.rl.execution import build_environment
    from openso101.rl.scene_geometry import robot_collision_extras, table_collision_geometry
    from openso101.rl.vision_distillation import action_mapping
    from openso101.robots import SO101_SIM_JOINT_NAMES
    from openso101.tasks.shared.grasp import _jaw_force_magnitude

    trace = []
    physics = []

    class TaskRecorder(RecorderTerm):
        def record_post_step(self):
            runtime = self._env
            robot = runtime.scene["robot"]
            obj = runtime.scene["object"]
            object_position, object_rotation = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                                                         obj.data.root_pos_w, obj.data.root_quat_w)
            grip_position, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                                         runtime.scene["ee_frame"].data.target_pos_w[:, 0])
            values = {"joint_position": robot.data.joint_pos[:, ids], "joint_velocity": robot.data.joint_vel[:, ids],
                      "gravity_compensation": robot.root_physx_view.get_gravity_compensation_forces()[:, ids],
                      "joint_targets": torch.cat([runtime.action_manager.get_term(name).processed_actions
                                                   for name in runtime.action_manager.active_terms], dim=-1),
                      "object_position_root": object_position, "grasp_position_root": grip_position,
                      "object_quaternion_root": object_rotation,
                      "jaw_forces": torch.stack([_jaw_force_magnitude(runtime.scene[name])
                                                for name in ("gripper_jaw_contact", "moving_jaw_contact")], dim=-1),
                      "success": runtime.termination_manager.get_term("success"),
                      "terminated": runtime.reset_terminated, "truncated": runtime.reset_time_outs,
                      "phase": phase.clone(), "active": active.clone()}
            values["path_cursor"] = path_cursor.clone()
            values["desired_joint_position"] = desired.clone()
            if pick_place:
                command = runtime.command_manager.get_term("object_pose")
                values["task_stage"] = command.stage.clone()
                values["placement_hold_seconds"] = command.placement_hold_seconds.clone()
            values["jaw_net_force_vectors"] = torch.stack([
                runtime.scene[name].data.net_forces_w.sum(dim=1)
                for name in ("gripper_jaw_contact", "moving_jaw_contact")], dim=1)
            values["jaw_object_force_vectors"] = torch.stack([
                runtime.scene[name].data.force_matrix_w.sum(dim=(1, 2))
                for name in ("gripper_jaw_contact", "moving_jaw_contact")], dim=1)
            if args.with_cameras:
                for name in ("overhead_camera", "wrist_camera"):
                    rgb = runtime.scene[name].data.output["rgb"][0, :, :, :3]
                    if rgb.shape != (args.camera_resolution, args.camera_resolution, 3) or rgb.dtype != torch.uint8:
                        raise ValueError("原生相机的实际图像形状或数据类型不一致")
                    values[f"cameras/{name}"] = rgb
            if any(not torch.isfinite(value).all() for value in values.values()):
                raise RuntimeError("任务检查产生无效状态")
            trace.append({name: value.cpu().numpy().copy() for name, value in values.items()})
            return None, None

        def record_post_physics_decimation_step(self):
            physics.append(self._env.scene["robot"].root_physx_view.get_dof_velocities()[:, ids].cpu().numpy().copy())
            return None, None

    @configclass
    class TaskRecorderCfg(RecorderManagerBaseCfg):
        dataset_export_dir_path = str(args.output / "recorder")
        dataset_export_mode = DatasetExportMode.EXPORT_NONE
        task = RecorderTermCfg(class_type=TaskRecorder)

    args.recorder_cfg = TaskRecorderCfg()
    env = build_environment(args, training=True)
    env.reset()
    runtime = env.unwrapped
    robot = runtime.scene["robot"]
    mappings = action_mapping(runtime)
    ids = [robot.joint_names.index(name) for name in SO101_SIM_JOINT_NAMES]
    phase = torch.full((args.num_envs,), -1, dtype=torch.long, device=runtime.device)
    path_cursor = torch.zeros_like(phase)
    active = torch.ones(args.num_envs, dtype=torch.bool, device=runtime.device)

    def control_actions(desired):
        if args.task_profile == "grasp_v3":
            return ((desired - robot.data.joint_pos[:, ids]) / .04).clamp(-1, 1)
        gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, ids]
        stiffness = robot.root_physx_view.get_dof_stiffnesses()[:, ids].to(runtime.device)
        if (stiffness <= 0).any() or not torch.isfinite(gravity).all():
            raise RuntimeError("绝对位置任务检查需要有效的重力保持力矩和 stiffness")
        compensated = desired + gravity / stiffness
        return torch.stack([(compensated[:, index] - item["offset"]) / item["scale"]
                            for index, item in enumerate(mappings)], dim=-1).clamp(-1, 1)

    settling_steps = 10 if args.task_profile == "grasp_v4" else 0
    settling_target = robot.data.joint_pos[:, ids].clone()
    settling_target[:, -1] = .8
    desired = settling_target
    for _ in range(settling_steps):
        _, _, terminated, truncated, _ = env.step(control_actions(settling_target))
        if (terminated | truncated).any():
            raise RuntimeError("规划准备过程提前终止")
    obj = runtime.scene["object"]
    object_position, object_rotation = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                                                 obj.data.root_pos_w, obj.data.root_quat_w)
    states = {"seed": args.seed, "task": args.task, "control_dt": runtime.step_dt, "physics_dt": runtime.physics_dt,
              "settling_steps": settling_steps,
              "joint_stiffness": robot.root_physx_view.get_dof_stiffnesses()[0, ids].cpu().tolist(),
              "effort_limits": robot.data.joint_effort_limits[0, ids].cpu().tolist(),
              "soft_joint_limits": robot.data.soft_joint_pos_limits[0, ids].cpu().tolist(),
              "environments": [{"environment": index,
                                "joint_position": robot.data.joint_pos[index, ids].cpu().tolist(),
                                "object_position_root": object_position[index].cpu().tolist(),
                                "object_quaternion_root": object_rotation[index].cpu().tolist(),
                                "goal_position_root": runtime.command_manager.get_command("object_pose")[index, :3].cpu().tolist()}
                               for index in range(args.num_envs)]}
    geometry = table_collision_geometry(runtime.sim.stage, "/World/envs/env_0/Table",
                                         robot.data.root_pos_w[0].cpu().numpy(), robot.data.root_quat_w[0].cpu().numpy())
    states["planner_physics"] = {
        "physical_joint_limits": robot.data.joint_pos_limits[0, ids].cpu().tolist(),
        "physics_dt": runtime.physics_dt, "nominal_stiffness": states["joint_stiffness"],
        "nominal_damping": robot.data.joint_damping[0, ids].cpu().tolist(), "effort_limits": states["effort_limits"],
        "table_geometry": geometry, "table_height_root": geometry["top_height_root"],
        "object_size": list(runtime.cfg.scene.object.spawn.size), "object_mass": runtime.cfg.scene.object.spawn.mass_props.mass,
        "robot_collision_extras": robot_collision_extras(runtime.sim.stage, "/World/envs/env_0/Robot"),
        "task_goal_radius": (runtime.command_manager.get_term("object_pose").cfg.advance_threshold
                             + runtime.command_manager.get_term("object_pose").cfg.object_contact_radius)
                            if pick_place else runtime.termination_manager.get_term_cfg("success").params["goal_radius"],
    }
    if pick_place:
        command = runtime.command_manager.get_term("object_pose")
        for index, item in enumerate(states["environments"]):
            item["carry_goal_position_root"] = command.goal_for_stage(1)[index].cpu().tolist()
            item["place_goal_position_root"] = command.goal_for_stage(2)[index].cpu().tolist()
    body_positions, body_quaternions = subtract_frame_transforms(
        robot.data.root_pos_w[:, None].expand_as(robot.data.body_pos_w),
        robot.data.root_quat_w[:, None].expand_as(robot.data.body_quat_w),
        robot.data.body_pos_w, robot.data.body_quat_w,
    )
    initial_physics = {
        "joint_position": robot.data.joint_pos[:, ids], "joint_velocity": robot.data.joint_vel[:, ids],
        "object_position_root": object_position, "object_quaternion_root": object_rotation,
        "object_linear_velocity_root": quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w),
        "object_angular_velocity_root": quat_apply_inverse(robot.data.root_quat_w, obj.data.root_ang_vel_w),
        "robot_body_position_root": body_positions, "robot_body_quaternion_root": body_quaternions,
        "scene_gravity": torch.tensor(list(robot.data._physics_sim_view.get_gravity())),
    }
    for name in ("joint_stiffness", "joint_damping", "joint_armature", "joint_friction_coeff", "joint_vel_limits"):
        initial_physics[name] = getattr(robot.data, name)[:, ids]
    for asset_name, asset in (("robot", robot), ("object", obj)):
        for field, getter in (("body_mass", "masses"), ("body_inertia", "inertias"), ("body_com", "coms")):
            initial_physics[f"{asset_name}_{field}"] = getattr(asset.root_physx_view, f"get_{getter}")()
    if any(not torch.isfinite(value).all() for value in initial_physics.values()):
        raise RuntimeError("原生初始物理参数包含无效数值")
    physics_path = args.output / "initial_physics.hdf5"
    with h5py.File(physics_path, "x") as stream:
        for name, value in initial_physics.items():
            stream.create_dataset(name, data=value.detach().cpu().numpy()[None])
    states["physics_recording"] = {
        "robot_body_names": list(robot.body_names), "inertia_frame": "body_prim_at_center_of_mass",
        "inertia_matrix_order": "column_major", "com_pose_frame": "body_prim", "com_quaternion_order": "xyzw",
    }
    states["quaternion_order"] = "wxyz"
    states["task_parameters"] = runtime.termination_manager.get_term_cfg("success").params.copy()
    states["task_parameters"].pop("command_name", None)
    states["task_reference_height_root"] = float(runtime.scene.env_origins[0, 2] - robot.data.root_pos_w[0, 2])
    states["initial_physics_sha256"] = digest(physics_path)
    states_path = args.output / "initial_states.json"
    with states_path.open("x") as stream:
        json.dump(states, stream, indent=2)
    plan_path = args.output / "plan.json"
    with (args.output / "planning.log").open("x") as log:
        command = [str(args.planner_python), "scripts/plan_grasp_task.py", "--states", str(states_path),
                   "--robot-model", str(args.robot_model), "--output", str(plan_path)]
        if args.task_profile == "grasp_v4":
            command.extend(["--collision-bundle", "outputs/rl_progress/gripper_collision"])
        subprocess.run(command,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    plan = json.loads(plan_path.read_text())
    if plan["states_sha256"] != digest(states_path):
        raise ValueError("规划与实际初始状态不一致")
    targets = torch.tensor([[item["joint_position"] for item in environment["targets"]]
                            for environment in plan["environments"]], device=runtime.device)
    phase.zero_()
    path_cursor.zero_()
    paths = (torch.tensor([[item["path_joint_positions"] for item in environment["targets"]]
                           for environment in plan["environments"]], device=runtime.device)
             if args.task_profile == "grasp_v4" else None)
    if paths is not None:
        starts = torch.cat((torch.tensor([item["joint_position"][:5] for item in states["environments"]],
                                         device=runtime.device)[:, None], paths[:, :-1, -1]), dim=1)
        paths = torch.cat((starts.unsqueeze(2), paths), dim=2)
        lengths = torch.cat((torch.zeros_like(paths[:, :, :1, 0]),
                             torch.abs(paths[:, :, 1:] - paths[:, :, :-1]).amax(dim=-1).cumsum(dim=-1)), dim=-1)
        path_distance = torch.zeros(args.num_envs, device=runtime.device)
    finished_success = torch.zeros_like(active)
    contact_hold_steps = torch.zeros_like(phase)
    for step in range(runtime.max_episode_length - settling_steps):
        target_index = (phase - (phase >= 2).long()).clamp(0, targets.shape[1] - 1)
        arm_target = targets[torch.arange(args.num_envs, device=runtime.device), target_index]
        if paths is not None:
            rows = torch.arange(args.num_envs, device=runtime.device)
            selected_paths, selected_lengths = paths[rows, target_index], lengths[rows, target_index]
            tracking_error = (robot.data.joint_pos[:, ids[:5]] - desired[:, :5]).abs().amax(dim=-1)
            moving = (phase != 2) & active & (tracking_error < .08)
            if pick_place:
                moving &= phase < 6
            path_distance = torch.minimum(path_distance + moving * .75 * runtime.step_dt, selected_lengths[:, -1])
            holding = (phase == 2) | ((phase >= 6) if pick_place else torch.zeros_like(active))
            distance = torch.where(holding, selected_lengths[:, -1], path_distance)
            upper = torch.searchsorted(selected_lengths.contiguous(), distance[:, None].contiguous(), right=True)[:, 0]
            upper = upper.clamp(1, paths.shape[2] - 1)
            fraction = ((distance - selected_lengths[rows, upper - 1]) /
                        (selected_lengths[rows, upper] - selected_lengths[rows, upper - 1]).clamp_min(1e-8))
            arm_target = torch.lerp(selected_paths[rows, upper - 1], selected_paths[rows, upper], fraction[:, None])
            path_cursor = upper - 1
        jaw_target = torch.where((phase < 2) | ((phase >= 6) if pick_place else torch.zeros_like(active)), .8, 0.).unsqueeze(-1)
        desired = torch.cat((arm_target, jaw_target), dim=-1)
        actions = control_actions(desired)
        actions[~active] = 0
        _, _, terminated, truncated, _ = env.step(actions)
        sample = trace[-1]
        errors = torch.linalg.vector_norm(torch.as_tensor(sample["joint_position"], device=runtime.device)[:, :5]
                                         - arm_target, dim=-1)
        advanced = (phase < 2) & (errors < .06) & active
        if paths is not None:
            reached_sample = (errors < .015) & active & (phase != 2)
            advanced = (phase < 2) & reached_sample & (path_distance >= selected_lengths[:, -1])
            if pick_place:
                command = runtime.command_manager.get_term("object_pose")
                path_finished = reached_sample & (path_distance >= selected_lengths[:, -1])
                advanced |= (phase == 3) & path_finished & (command.stage >= 1)
                advanced |= (phase == 4) & path_finished & (command.stage >= 2)
                object_root = torch.as_tensor(sample["object_position_root"], device=runtime.device)
                at_place = torch.linalg.vector_norm(object_root - command.goal_for_stage(2), dim=-1) <= .03
                advanced |= (phase == 5) & path_finished & at_place
        forces = torch.as_tensor(sample["jaw_forces"], device=runtime.device)
        contact_hold_steps = torch.where((phase == 2) & (forces > .5).all(dim=-1), contact_hold_steps + 1, 0)
        advanced |= (phase == 2) & (contact_hold_steps * runtime.step_dt >= .1) & active
        path_cursor = torch.where(advanced, 0, path_cursor)
        if paths is not None:
            path_distance = torch.where(advanced, 0., path_distance)
        phase += advanced.long()
        finished_success |= torch.as_tensor(sample["success"], device=runtime.device) & active
        active &= ~(terminated | truncated)
        if not active.any():
            break
    if not trace or len(physics) != len(trace) * runtime.cfg.decimation:
        raise RuntimeError("实际任务轨迹的步骤数量不完整")
    trajectory = args.output / "trajectory.hdf5"
    with h5py.File(trajectory, "x") as stream:
        for name in trace[0]:
            stream.create_dataset(name, data=np.stack([item[name] for item in trace]),
                                  compression="gzip" if name.startswith("cameras/") else None)
        stream.create_dataset("physics_steps/joint_velocity", data=np.asarray(physics))
        stream.attrs["control_dt"] = runtime.step_dt
        stream.attrs["physics_dt"] = runtime.physics_dt
    records = []
    for index in range(args.num_envs):
        samples = [item for item in trace if item["active"][index]]
        forces = np.stack([item["jaw_forces"][index] for item in samples])
        heights = np.asarray([item["object_position_root"][index, 2] for item in samples])
        records.append({"environment": index, "success": bool(finished_success[index]),
                        "control_steps": len(samples), "maximum_phase": int(max(item["phase"][index] for item in samples)),
                        "maximum_jaw_forces_n": forces.max(axis=0).tolist(),
                        "bilateral_contact_steps": int((forces > .5).all(axis=-1).sum()),
                        "maximum_object_height_root_m": float(heights.max())})
    report = {"status": "native_task_completed", "task": args.task, "task_profile": args.task_profile,
              "environment_mode": args.environment_mode, "seed": args.seed, "environments": records,
              "successes": int(finished_success.sum()), "episodes": args.num_envs,
              "controller": "scripted_IK_measured_reference_delta" if args.task_profile == "grasp_v3"
                            else "scripted_IK_gravity_compensated_joint_targets", "control_dt": runtime.step_dt,
              "physics_dt": runtime.physics_dt, "maximum_physics_speed_rad_s": float(np.abs(physics).max()),
              "settling_steps": settling_steps, "bilateral_hold_before_lift_s": .1,
              "planned_joint_speed_rad_s": .75 if paths is not None else None,
              "planned_phases": [item["phase"] for item in plan["environments"][0]["targets"]],
              "path_tracking_pause_error_rad": .08 if paths is not None else None,
              "states_sha256": digest(states_path), "plan_sha256": digest(plan_path),
              "initial_physics_sha256": digest(physics_path),
              "trace_sha256": digest(trajectory), "source_sha256": digest(Path(__file__)),
              "profile_sha256": digest(Path(f"src/openso101/tasks/shared/{args.task_profile}.py")),
              "rl_policy_success_verified": False}
    gravity = np.stack([item["gravity_compensation"] for item in trace])
    holding_budget = (np.asarray(states["joint_stiffness"]) * .04 if args.task_profile == "grasp_v3"
                      else np.asarray(states["effort_limits"]))
    report["gravity_diagnostics"] = {
        "maximum_required_holding_effort_nm": np.abs(gravity).max(axis=(0, 1)).tolist(),
        "stationary_effort_budget_nm": holding_budget.tolist(),
        "frames_exceeding_stationary_effort_budget": (np.abs(gravity) > holding_budget).sum(axis=(0, 1)).tolist(),
        "measured_frames": len(trace) * args.num_envs,
    }
    if args.with_cameras:
        import av
        from fractions import Fraction

        fps = Fraction(1 / runtime.step_dt).limit_denominator(10000)
        selected = [item for item in trace if item["active"][0]]
        video_path = args.output / "native_grasp.mp4"
        with av.open(str(video_path), "w") as movie:
            video = movie.add_stream("libx264", rate=fps)
            video.width, video.height = args.camera_resolution * 2, args.camera_resolution
            video.pix_fmt = "yuv420p"
            video.options = {"crf": "18", "preset": "fast"}
            for item in selected:
                image = np.concatenate([item[f"cameras/{name}"] for name in ("overhead_camera", "wrist_camera")], axis=1)
                if np.std(image) <= 0:
                    raise ValueError("原生视频的相机图像缺少变化")
                for packet in video.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    movie.mux(packet)
            for packet in video.encode():
                movie.mux(packet)
        with av.open(str(video_path)) as movie:
            decoded = sum(1 for _ in movie.decode(video=0))
            if decoded != len(selected) or movie.streams.video[0].average_rate != fps:
                raise ValueError("原生视频的编码帧数或频率不一致")
        report["video"] = {"path": str(video_path.resolve()), "sha256": digest(video_path),
                           "frames": decoded, "fps": float(fps), "environment": 0,
                           "resolution": [args.camera_resolution * 2, args.camera_resolution],
                           "camera_order": ["overhead_camera", "wrist_camera"],
                           "scripted_task_success": bool(finished_success[0]), "rl_policy_success_verified": False}
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)
finally:
    if env is not None:
        env.close()
app.close()
