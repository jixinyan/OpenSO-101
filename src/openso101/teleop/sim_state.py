from typing import Any


_REPLAY_COMMAND_FIELDS = (
    "stage", "goal_pos_b", "goal_pos_w", "cube_spawn_xy_b", "placement_hold_seconds", "pose_command_b",
)


def _tensor_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _collect_replay_sim_state(unwrapped_env, scene, *, include_cohort=False) -> dict[str, Any]:
    import numpy as np

    sim_state: dict[str, Any] = {"environment_origin": _tensor_to_numpy(scene.env_origins[0])}
    if getattr(unwrapped_env.cfg, "scene_spec", None) is not None:
        from openso101.scenes.isaaclab.runtime import bddl_trackers, scene_jaw_forces, scene_states
        from openso101.rl.student import student_goal
        import torch

        sim_state["scene_entity_states"] = _tensor_to_numpy(scene_states(unwrapped_env)[0])
        forces = scene_jaw_forces(unwrapped_env)
        sim_state["scene_jaw_forces"] = _tensor_to_numpy(torch.stack(list(forces.values()), dim=1)[0])
        trackers = bddl_trackers(unwrapped_env)
        if trackers:
            sim_state["scene_bddl_hold"] = np.asarray(trackers[0].elapsed)
        for name in ("_scene_hold_seconds", "_scene_success", "_scene_program_phase", "_scene_program_hold"):
            if hasattr(unwrapped_env, name):
                sim_state[name.removeprefix("_")] = _tensor_to_numpy(getattr(unwrapped_env, name)[0])
        sim_state["task_goal_root"] = _tensor_to_numpy(student_goal(unwrapped_env)[0])
        return sim_state
    for name in ("object", "cube_top", "cube_bottom"):
        if name in scene.rigid_objects:
            sim_state[f"{name}_root_state"] = _tensor_to_numpy(scene[name].data.root_state_w[0])
            if include_cohort and name == "object":
                sim_state["cohort_object_root_state"] = _tensor_to_numpy(scene[name].data.root_state_w)
    if "cube_top" in scene.rigid_objects:
        from openso101.teleop.success import task_success_vector

        task_success_vector(unwrapped_env)
        sim_state["cube_top_was_lifted"] = _tensor_to_numpy(unwrapped_env._cube_top_was_lifted[0])
        sim_state["task_episode_step"] = _tensor_to_numpy(unwrapped_env.episode_length_buf[0])
    if include_cohort:
        import torch

        robot = scene["robot"]
        ids = _replay_robot_joint_indices(robot)
        sim_state.update({
            "cohort_environment_origins": _tensor_to_numpy(scene.env_origins),
            "cohort_joint_position": _tensor_to_numpy(robot.data.joint_pos[:, ids]),
            "cohort_joint_velocity": _tensor_to_numpy(robot.data.joint_vel[:, ids]),
            "cohort_joint_targets": _tensor_to_numpy(torch.cat([
                unwrapped_env.action_manager.get_term(name).processed_actions
                for name in unwrapped_env.action_manager.active_terms], dim=-1)),
            "cohort_policy_actions": _tensor_to_numpy(unwrapped_env.action_manager.action),
        })
    if "object_pose" in unwrapped_env.command_manager.active_terms:
        from openso101.rl.student import student_goal

        sim_state["task_goal_root"] = _tensor_to_numpy(student_goal(unwrapped_env)[0])
        command = unwrapped_env.command_manager.get_term("object_pose")
        for field in _REPLAY_COMMAND_FIELDS:
            if hasattr(command, field):
                sim_state[f"command_{field}"] = _tensor_to_numpy(getattr(command, field)[0])
                if include_cohort:
                    sim_state[f"cohort_command_{field}"] = _tensor_to_numpy(getattr(command, field))
    if "success" in unwrapped_env.termination_manager.active_terms:
        success = unwrapped_env.termination_manager.get_term_cfg("success").func
        if hasattr(success, "hold_seconds"):
            sim_state["task_hold_seconds"] = _tensor_to_numpy(success.hold_seconds[0])
            if include_cohort:
                sim_state["cohort_task_hold_seconds"] = _tensor_to_numpy(success.hold_seconds)
        sim_state["task_episode_step"] = _tensor_to_numpy(unwrapped_env.episode_length_buf[0])
        if include_cohort:
            sim_state["cohort_task_episode_step"] = _tensor_to_numpy(unwrapped_env.episode_length_buf)
    return sim_state


def _replay_to_tensor_like(values, reference):
    import torch

    return torch.as_tensor(values, device=reference.device, dtype=reference.dtype)


def _replay_robot_joint_indices(robot) -> list[int]:
    from openso101.robots import SO101_SIM_JOINT_NAMES

    joint_names = list(robot.joint_names)
    return [joint_names.index(joint_name) for joint_name in SO101_SIM_JOINT_NAMES]


def _replay_set_robot_proprio(scene, qpos, qvel) -> None:
    robot = scene["robot"]
    joint_ids = _replay_robot_joint_indices(robot)
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    joint_pos[0, joint_ids] = _replay_to_tensor_like(qpos, joint_pos[0, joint_ids])
    joint_vel[0, joint_ids] = _replay_to_tensor_like(qvel, joint_vel[0, joint_ids])
    robot.write_joint_position_to_sim(joint_pos)
    robot.write_joint_velocity_to_sim(joint_vel)
    robot.set_joint_position_target(joint_pos)


def _replay_optional_frame(h5, dataset_name: str, frame_index: int):
    import numpy as np

    if dataset_name not in h5:
        return None
    return np.asarray(h5[dataset_name][frame_index])


def _replay_restore_sim_state_from_episode(unwrapped_env, scene, h5, frame_index: int) -> None:
    import numpy as np

    if getattr(unwrapped_env, "_replay_cohort", False):
        origins = h5["sim/cohort_environment_origins"][frame_index]
        if not np.allclose(_tensor_to_numpy(scene.env_origins), origins, atol=1e-6, rtol=0):
            raise ValueError("并行回放需要使用来源环境布局")
        robot = scene["robot"]
        ids = _replay_robot_joint_indices(robot)
        positions, velocities = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
        positions[:, ids] = _replay_to_tensor_like(h5["sim/cohort_joint_position"][frame_index], positions[:, ids])
        velocities[:, ids] = _replay_to_tensor_like(h5["sim/cohort_joint_velocity"][frame_index], velocities[:, ids])
        robot.write_joint_position_to_sim(positions)
        robot.write_joint_velocity_to_sim(velocities)
        robot.set_joint_position_target(positions)
        obj = scene["object"]
        obj.write_root_state_to_sim(_replay_to_tensor_like(h5["sim/cohort_object_root_state"][frame_index], obj.data.root_state_w))
        command = unwrapped_env.command_manager.get_term("object_pose")
        for field in _REPLAY_COMMAND_FIELDS:
            key = f"sim/cohort_command_{field}"
            if hasattr(command, field):
                if key not in h5:
                    raise ValueError(f"并行采集缺少任务命令: {key}")
                target = getattr(command, field)
                target[:] = _replay_to_tensor_like(h5[key][frame_index], target)
        success = unwrapped_env.termination_manager.get_term_cfg("success").func
        if hasattr(success, "hold_seconds"):
            success.hold_seconds[:] = _replay_to_tensor_like(h5["sim/cohort_task_hold_seconds"][frame_index], success.hold_seconds)
        unwrapped_env.episode_length_buf[:] = _replay_to_tensor_like(
            h5["sim/cohort_task_episode_step"][frame_index], unwrapped_env.episode_length_buf)
        return

    _replay_set_robot_proprio(
        scene, qpos=np.asarray(h5["observations/qpos"][frame_index], dtype=np.float32),
        qvel=np.asarray(h5["observations/qvel"][frame_index], dtype=np.float32),
    )
    if getattr(unwrapped_env.cfg, "scene_spec", None) is not None:
        states = h5["sim/scene_entity_states"][frame_index]
        entities = unwrapped_env.cfg.scene_spec.entities
        if states.shape != (len(entities), 13) or not np.isfinite(states).all():
            raise ValueError("采集实体状态格式错误")
        for index, entity in enumerate(entities):
            if entity.dynamic:
                obj = scene[entity.entity_id]
                state = _replay_to_tensor_like(states[index][None, ...], obj.data.root_state_w)
                state[:, :3] += scene.env_origins
                obj.write_root_state_to_sim(state)
        for name in ("scene_hold_seconds", "scene_success", "scene_program_phase", "scene_program_hold"):
            value = _replay_optional_frame(h5, f"sim/{name}", frame_index)
            if value is not None:
                import torch

                dtype = torch.int64 if name == "scene_program_phase" else torch.bool if name == "scene_success" else torch.float32
                restored = torch.as_tensor(value, device=unwrapped_env.device, dtype=dtype).expand(unwrapped_env.num_envs).clone()
                setattr(unwrapped_env, "_" + name, restored)
            elif hasattr(unwrapped_env, "_" + name):
                getattr(unwrapped_env, "_" + name).zero_()
        if hasattr(unwrapped_env, "_scene_hold_seconds"):
            unwrapped_env._scene_success_step = -1
        from openso101.scenes.isaaclab.runtime import bddl_trackers

        trackers = bddl_trackers(unwrapped_env)
        if trackers:
            held = _replay_optional_frame(h5, "sim/scene_bddl_hold", frame_index)
            if held is None:
                raise ValueError("BDDL 场景回放需要保存的任务保持时间")
            for tracker in trackers:
                progress = tracker.snapshot().model_copy(update={"held_seconds": float(held)})
                tracker.restore(progress)
        return

    from openso101.teleop.state_records import replay_root_state

    for name in ("object", "cube_top", "cube_bottom"):
        if name in scene.rigid_objects:
            obj = scene[name]
            values = replay_root_state(h5[f"sim/{name}_root_state"][frame_index],
                                       h5["sim/environment_origin"][frame_index],
                                       _tensor_to_numpy(scene.env_origins[0]))
            obj.write_root_state_to_sim(_replay_to_tensor_like(values[None, ...], obj.data.root_state_w))
    if "cube_top" in scene.rigid_objects:
        import torch

        unwrapped_env._cube_top_was_lifted = torch.as_tensor(
            h5["sim/cube_top_was_lifted"][frame_index], device=unwrapped_env.device,
            dtype=torch.bool).expand(unwrapped_env.num_envs).clone()
    command_values = {
        field: _replay_optional_frame(h5, f"sim/command_{field}", frame_index) for field in _REPLAY_COMMAND_FIELDS
    }
    if any(value is not None for value in command_values.values()):
        command = unwrapped_env.command_manager.get_term("object_pose")
        for field, value in command_values.items():
            if value is not None:
                target = getattr(command, field)
                restored = _replay_to_tensor_like(value, target[0])
                if field == "goal_pos_w":
                    origin = _replay_optional_frame(h5, "sim/environment_origin", frame_index)
                    if origin is not None:
                        restored += scene.env_origins[0] - _replay_to_tensor_like(origin, scene.env_origins[0])
                target[0] = restored
    hold = _replay_optional_frame(h5, "sim/task_hold_seconds", frame_index)
    if hold is not None:
        success = unwrapped_env.termination_manager.get_term_cfg("success").func
        success.hold_seconds[0] = _replay_to_tensor_like(hold, success.hold_seconds[0])
    episode_step = _replay_optional_frame(h5, "sim/task_episode_step", frame_index)
    if episode_step is not None:
        unwrapped_env.episode_length_buf[0] = _replay_to_tensor_like(episode_step, unwrapped_env.episode_length_buf[0])
