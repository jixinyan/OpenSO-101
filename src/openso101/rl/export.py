import json
import subprocess
from pathlib import Path

from .config import CheckpointMeta, digest


def export(args):
    folder = Path(args.checkpoint).resolve()
    meta = CheckpointMeta.read(folder)
    if meta.task_id != args.task or meta.config.backend != "rsl_rl" or meta.config.algo != "ppo":
        raise ValueError("policy 导出需要请求任务的 RSL PPO checkpoint")
    if args.task not in ("OpenSO101-Lift-v0", "OpenSO101-PickPlace-v0"):
        raise ValueError("policy 导出支持 Lift 与 PickPlace")
    if args.validation_steps <= 0 or args.num_envs <= 0:
        raise ValueError("validation_steps 与 num_envs 必须为正数")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=args.headless).app
    env = None
    try:
        import h5py
        import numpy as np
        import torch
        from isaaclab.utils.math import subtract_frame_transforms
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit
        from rsl_rl.runners import OnPolicyRunner

        from openso101.robots.so101.constants import SO101_SIM_JOINT_NAMES
        from openso101.tasks.shared.grasp import _jaw_force_magnitude, object_grasped_by_jaws

        from .execution import build_environment
        from .portable import PortablePolicy
        from .vision_distillation import action_mapping

        args.seed = meta.config.seed
        args.task_profile = meta.task_profile
        env = build_environment(args, training=False)
        unwrapped = env.unwrapped
        config = json.loads((folder / "backend.json").read_text())
        runner = OnPolicyRunner(RslRlVecEnvWrapper(env), config, log_dir=None, device=unwrapped.device)
        runner.load(str(folder / meta.checkpoint), load_optimizer=False)
        actor = runner.alg.policy.eval()
        export_policy_as_jit(actor, actor.actor_obs_normalizer, str(output), filename="policy.pt")
        robot = unwrapped.scene["robot"]
        joint_ids = [robot.joint_names.index(name) for name in SO101_SIM_JOINT_NAMES]
        observation_terms = [
            {"name": name, "size": int(np.prod(shape))}
            for name, shape in zip(unwrapped.observation_manager.active_terms["policy"],
                                   unwrapped.observation_manager.group_obs_term_dim["policy"], strict=True)
        ]
        expected_names = ["joint_pos", "joint_vel", "object_position", "target_object_position", "grasp_state", "actions"]
        if [term["name"] for term in observation_terms] != expected_names:
            raise ValueError("policy 观测定义不支持 portable 导出")
        if robot.num_joints != 6:
            raise ValueError("policy 导出需要六个 SO-101 关节")
        metadata = {
            "schema_version": 1, "task_id": meta.task_id, "task_profile": meta.task_profile, "training_git_sha": meta.git_sha,
            "export_git_sha": revision, "checkpoint_sha256": digest(folder / meta.checkpoint),
            "files": {"policy.pt": digest(output / "policy.pt")}, "control_dt": unwrapped.step_dt,
            "physics_dt": unwrapped.physics_dt, "joint_names": list(SO101_SIM_JOINT_NAMES),
            "observation_joint_names": robot.joint_names,
            "physical_joint_limits": robot.data.joint_pos_limits[0, joint_ids].tolist(),
            "default_joint_positions": robot.data.default_joint_pos[0].tolist(),
            "default_joint_velocities": robot.data.default_joint_vel[0].tolist(),
            "nominal_stiffness": robot.data.default_joint_stiffness[0, joint_ids].tolist(),
            "nominal_damping": robot.data.default_joint_damping[0, joint_ids].tolist(),
            "effort_limits": robot.data.joint_effort_limits[0, joint_ids].tolist(),
            "table_height_root": float(unwrapped.scene.env_origins[0, 2] - robot.data.root_pos_w[0, 2]),
            "object_size": list(unwrapped.cfg.scene.object.spawn.size),
            "object_mass": unwrapped.cfg.scene.object.spawn.mass_props.mass,
            "episode_length_s": unwrapped.cfg.episode_length_s,
            "observation_terms": observation_terms, "normalization": "embedded_in_policy.pt",
            "position_frame": "robot_root", "quaternion_order": "wxyz", "joint_unit": "radian",
            "last_action": "raw_policy_action", "action_mapping": action_mapping(unwrapped),
            "grasp_force_threshold_newtons": 0.5,
            "reward_terms": [{"name": name, "weight": unwrapped.reward_manager.get_term_cfg(name).weight}
                             for name in unwrapped.reward_manager.active_terms],
            "hardware_observations_required": ["object_position_root", "goal_root", "grasp_state"],
        }
        if args.task == "OpenSO101-PickPlace-v0":
            command_cfg = unwrapped.command_manager.get_term("object_pose").cfg
            metadata["task_parameters"] = {
                "carry_height": command_cfg.carry_height, "place_goal": list(command_cfg.place_goal),
                "advance_threshold": command_cfg.advance_threshold, "object_contact_radius": command_cfg.object_contact_radius,
                "place_radius": 0.03, "linear_speed_max": 0.02, "angular_speed_max": 0.1,
                "jaw_open_min": 0.4,
                "settle_seconds": unwrapped.termination_manager.get_term_cfg("success").params["settle_seconds"],
            }
        else:
            metadata["task_parameters"] = unwrapped.termination_manager.get_term_cfg("success").params.copy()
            metadata["task_parameters"].pop("command_name")
        (output / "policy.json").write_text(json.dumps(metadata, indent=2))
        portable = PortablePolicy(output)
        observation, _ = env.reset()
        maximum_errors = {"observation": 0., "policy_action": 0., "processed_targets": 0.}
        buffers = {name: [] for name in (
            "joint_position", "joint_velocity", "object_position_root", "object_quaternion_root", "goal_root", "jaw_forces",
            "ee_object_distance", "gripper_position_root", "gripper_quaternion_root",
            "raw_action", "joint_targets", "weighted_reward", "terminated", "truncated",
        )}
        for _ in range(args.validation_steps):
            with torch.inference_mode():
                obj = unwrapped.scene["object"]
                object_root, object_quaternion = subtract_frame_transforms(
                    robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w, obj.data.root_quat_w,
                )
                goal = unwrapped.command_manager.get_command("object_pose")
                grasp = object_grasped_by_jaws(unwrapped).float().unsqueeze(-1)
                rebuilt = portable.observation(robot.data.joint_pos, robot.data.joint_vel, object_root, goal,
                                               grasp, unwrapped.action_manager.action)
                actions = actor.act_inference(observation)
                exported_actions = portable.predict(rebuilt)
                decoded = portable.joint_targets(exported_actions, enforce_limits=False)
                errors = {"observation": float((rebuilt - observation["policy"].cpu()).abs().max()),
                          "policy_action": float((exported_actions - actions.cpu()).abs().max())}
                for name, error in errors.items():
                    maximum_errors[name] = max(maximum_errors[name], error)
                    if error > 1e-5:
                        raise RuntimeError(f"policy 数值检查失败：{name}={error}")
                before_step = {
                    "joint_position": robot.data.joint_pos[:, joint_ids], "joint_velocity": robot.data.joint_vel[:, joint_ids],
                    "object_position_root": object_root, "object_quaternion_root": object_quaternion, "goal_root": goal,
                    "jaw_forces": torch.stack([_jaw_force_magnitude(unwrapped.scene[name])
                                               for name in ("gripper_jaw_contact", "moving_jaw_contact")], dim=-1),
                    "ee_object_distance": torch.linalg.vector_norm(
                        obj.data.root_pos_w - unwrapped.scene["ee_frame"].data.target_pos_w[:, 0], dim=-1),
                    "raw_action": actions, "joint_targets": decoded,
                }
                gripper_id = robot.body_names.index("gripper")
                gripper_position, gripper_quaternion = subtract_frame_transforms(
                    robot.data.root_pos_w, robot.data.root_quat_w,
                    robot.data.body_pos_w[:, gripper_id], robot.data.body_quat_w[:, gripper_id],
                )
                before_step.update(gripper_position_root=gripper_position, gripper_quaternion_root=gripper_quaternion)
                for name, value in before_step.items():
                    buffers[name].append(value.detach().cpu().numpy().copy())
                observation, _, terminated, truncated, _ = env.step(actions)
                processed_by_joint = {}
                for name in unwrapped.action_manager.active_terms:
                    term = unwrapped.action_manager.get_term(name)
                    ids = list(range(robot.num_joints)) if isinstance(term._joint_ids, slice) else list(term._joint_ids)
                    processed_by_joint.update({robot.joint_names[joint_id]: term.processed_actions[:, index]
                                               for index, joint_id in enumerate(ids)})
                processed = torch.stack([processed_by_joint[name] for name in SO101_SIM_JOINT_NAMES], dim=-1)
                target_error = float((processed.cpu() - decoded).abs().max())
                maximum_errors["processed_targets"] = max(maximum_errors["processed_targets"], target_error)
                if target_error > 1e-5:
                    raise RuntimeError("policy 动作转换检查失败")
                buffers["weighted_reward"].append((unwrapped.reward_manager._step_reward * unwrapped.step_dt).cpu().numpy().copy())
                buffers["terminated"].append(terminated.cpu().numpy().copy())
                buffers["truncated"].append(truncated.cpu().numpy().copy())
        arrays = {name: np.asarray(value) for name, value in buffers.items()}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError("policy 诊断轨迹包含非有限数值")
        with h5py.File(output / "isaac_validation.hdf5", "w") as trace:
            trace.attrs["state_sampling"] = "before_control_step"
            trace.attrs["reward_sampling"] = "transition_reward"
            trace.attrs["reward_terms"] = json.dumps(unwrapped.reward_manager.active_terms)
            trace.attrs["joint_names"] = json.dumps(list(SO101_SIM_JOINT_NAMES))
            for name, value in arrays.items():
                trace.create_dataset(name, data=value)
        jaw_target = arrays["joint_targets"][..., -1]
        closed = jaw_target < 0.4
        summary = {
            "status": "portable_policy_numerically_verified_in_isaac", "git_sha": revision,
            "validation_steps": args.validation_steps, "num_envs": unwrapped.num_envs,
            "maximum_errors": maximum_errors, "close_command_fraction": float(closed.mean()),
            "jaw_sign_changes": int(np.count_nonzero(closed[1:] != closed[:-1])),
            "dual_contact_fraction": float((arrays["jaw_forces"] > 0.5).all(axis=-1).mean()),
            "mean_ee_object_distance_m": float(arrays["ee_object_distance"].mean()),
            "weighted_reward_totals": dict(zip(unwrapped.reward_manager.active_terms,
                                               arrays["weighted_reward"].sum(axis=(0, 1)).tolist(), strict=True)),
            "trace_sha256": digest(output / "isaac_validation.hdf5"),
            "task_success_verified": False,
        }
        (output / "validation.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary))
    finally:
        if env is not None:
            env.close()
    app.close()
    return 0
