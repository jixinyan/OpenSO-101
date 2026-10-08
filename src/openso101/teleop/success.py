def task_success_vector(env, command_name: str = "object_pose"):
    if getattr(env.cfg, "scene_spec", None) is not None:
        from openso101.scenes.isaaclab.runtime import task_success

        return task_success(env)
    if "cube_top" in env.scene.rigid_objects and "cube_bottom" in env.scene.rigid_objects:
        from openso101.tasks.shared.rl_defaults import SO101_CONTROLLED_OBJECT_MIN_HEIGHT
        from openso101.tasks.stack.mdp.rewards import get_cube_top_was_lifted
        from openso101.tasks.stack.mdp.terminations import cubes_stacked_success

        was_lifted = get_cube_top_was_lifted(env)
        was_lifted |= env.scene["cube_top"].data.root_pos_w[:, 2] > SO101_CONTROLLED_OBJECT_MIN_HEIGHT
        return cubes_stacked_success(env)
    command = env.command_manager.get_term(command_name)
    if hasattr(command, "placement_hold_seconds"):
        from openso101.tasks.pick_place.mdp.terminations import released_at_place_goal

        return released_at_place_goal(env, command_name)
    from openso101.tasks.lift.mdp.terminations import lift_success_height_only
    from openso101.tasks.shared.rl_defaults import SO101_CONTROLLED_OBJECT_MIN_HEIGHT

    return lift_success_height_only(env, minimal_height=SO101_CONTROLLED_OBJECT_MIN_HEIGHT,
                                    goal_radius=0.05, command_name=command_name)


def task_success(env, command_name: str = "object_pose") -> bool:
    return bool(task_success_vector(env, command_name)[0])
