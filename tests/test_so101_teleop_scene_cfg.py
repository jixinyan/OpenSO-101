import pytest

pytest.importorskip("isaaclab_tasks")

import gymnasium as gym
import isaaclab.sim as sim_utils

import openso101.robots as robots
import openso101.tasks.pick_place  # noqa: F401
from openso101.robots.so101.so_arm101 import SO101_USD_TABLETOP_ROOT_Z


def test_so101_training_scene_uses_canonical_usd_robot_backend():
    env = gym.make("OpenSO101-PickPlace-v0")
    cfg = env.unwrapped.cfg
    env.close()

    assert cfg.scene.robot.spawn.usd_path.endswith("SO-ARM101-USD.usd")
    # RL spawn binds a high-friction material on the gripper colliders
    # (spawn_so101_usd_with_grip_friction). The aggressive variant that
    # mutated CollisionAPI was deleted; the current func is minimal.
    from openso101.robots.so101.so_arm101 import spawn_so101_usd_with_grip_friction
    assert cfg.scene.robot.spawn.func is spawn_so101_usd_with_grip_friction
    assert cfg.scene.robot.spawn.activate_contact_sensors is True
    assert cfg.scene.robot.spawn.articulation_props.fix_root_link is True
    assert cfg.scene.robot.init_state.pos == pytest.approx((0.0, 0.0, SO101_USD_TABLETOP_ROOT_Z))
    assert cfg.scene.robot.init_state.joint_pos["Jaw"] == pytest.approx(1.745)
    assert cfg.scene.ee_frame.prim_path == "{ENV_REGEX_NS}/Robot/base"
    assert cfg.scene.ee_frame.target_frames[0].prim_path == "{ENV_REGEX_NS}/Robot/gripper"
    assert cfg.scene.ee_frame.target_frames[0].offset.pos == pytest.approx((0.01, 0.0, -0.09))
    assert cfg.actions.arm_action.joint_names == ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
    assert cfg.actions.gripper_action.joint_names == ["Jaw"]
    assert cfg.scene.fixed_fingertip_pad is None
    assert cfg.scene.moving_fingertip_pad is None


def test_so101_pick_place_freezes_goal_at_air_carry_target():
    """RL pick-and-lift uses a single frozen air goal (sentinel design).

    The command is frozen at ``lock_stage=1`` so the goal is the carry target
    ``(place_goal.x, place_goal.y, carry_height)`` in the air — teleop's X/Y at
    carry height. ``__post_init__`` lowers the heights/place-z by the canonical
    USD tabletop root drop so they land at the intended world height.
    """
    import openso101.tasks.pick_place.mdp as mdp
    from openso101.tasks.pick_place import PickPlaceEnvCfg

    # Instantiate the cfg directly (no gym.make) so we don't allocate the
    # 4096-env scene just to inspect config fields. __post_init__ applies the
    # tabletop-root height offsets.
    cfg = PickPlaceEnvCfg()
    root_drop = -SO101_USD_TABLETOP_ROOT_Z

    # Single frozen goal at the air carry target.
    assert cfg.commands.object_pose.lock_stage == 1
    assert cfg.commands.object_pose.carry_height == pytest.approx(0.15 + root_drop)
    # lift_height + place_goal.z are kept consistent for teleop's lock_stage=2.
    assert cfg.commands.object_pose.lift_height == pytest.approx(0.10 + root_drop)
    assert cfg.commands.object_pose.place_goal == pytest.approx((0.24, -0.3, 0.0 + root_drop))

    # Sentinel delta-shaping reward terms replace the old 3-stage chain.
    assert cfg.rewards.pregrasp_approach.func is mdp.pregrasp_approach_shaping
    assert cfg.rewards.carry_to_goal.func is mdp.carry_to_goal_shaping
    assert cfg.rewards.grasp_hold.func is mdp.grasped_reward
    assert cfg.rewards.success_bonus.weight == pytest.approx(50.0)
    assert not hasattr(cfg.rewards, "stage_0_lift_goal")

    # Success requires reaching the goal AND a confirmed grasp.
    assert cfg.terminations.success.func is mdp.reached_goal_while_grasped


def test_so101_robot_public_exports_only_canonical_config():
    assert "SO_ARM101_CFG" in robots.__all__
    assert "SO_ARM101_UPSTREAM_USD_CFG" not in robots.__all__
    assert not hasattr(robots, "SO_ARM101_UPSTREAM_USD_CFG")


def test_pick_place_registry_has_no_separate_usd_ab_task_ids():
    assert gym.spec("OpenSO101-PickPlace-v0").kwargs["env_cfg_entry_point"].endswith(":PickPlaceEnvCfg")
    with pytest.raises(gym.error.NameNotFound):
        gym.spec("OpenSO101-USD-PickPlace-v0")


def test_so101_training_cube_uses_isaac_primitive_cube_collision():
    env = gym.make("OpenSO101-PickPlace-v0")
    cfg = env.unwrapped.cfg
    env.close()

    assert isinstance(cfg.scene.object.spawn, sim_utils.CuboidCfg)


def test_so101_teleop_scene_uses_canonical_usd_without_manual_fingertip_pads():
    env = gym.make("OpenSO101-PickPlace-v0", action_mode="teleop")
    cfg = env.unwrapped.cfg
    env.close()

    assert cfg.scene.robot.spawn.usd_path.endswith("SO-ARM101-USD.usd")
    # Teleop trusts the USD's authored colliders + friction verbatim
    # (Lior-style: liorbenhorin/lerobot_so101_teleop). The RL-only friction
    # binding spawn function (spawn_so101_usd_with_grip_friction) must NOT
    # be wired here — human-led teleop has its own friction compensation
    # via the leader-arm control loop.
    from openso101.robots.so101.so_arm101 import spawn_so101_usd_with_grip_friction
    assert cfg.scene.robot.spawn.func is not spawn_so101_usd_with_grip_friction
    assert cfg.scene.robot.spawn.activate_contact_sensors is False
    # Contact sensors must be stripped: the teleop robot has no contact
    # reporter API on its bodies, and no teleop reward consumes the signal.
    # Leaving them wired causes InteractiveScene to fail at init.
    assert cfg.scene.gripper_jaw_contact is None
    assert cfg.scene.moving_jaw_contact is None
    assert cfg.actions.joint_positions.joint_names == ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    assert cfg.scene.fixed_fingertip_pad is None
    assert cfg.scene.moving_fingertip_pad is None
    assert cfg.rewards is None
    assert cfg.terminations is None
    assert cfg.sim.dt == pytest.approx(1 / 120)
    assert cfg.sim.render_interval == cfg.decimation


def test_so101_teleop_uses_compliant_lior_style_actuators():
    """Teleop adopts liorbenhorin/lerobot_so101_teleop's compliant gains.

    Low stiffness with effort_limit_sim=30 across all joints matches real
    SO-101 servo bandwidth; lets the gripper pinch rather than hammer. Our
    earlier high-stiffness config (k=60, d=20 on the gripper) caused the
    jaws to slam shut and bounce small objects out before contact engaged.
    """
    env = gym.make("OpenSO101-PickPlace-v0", action_mode="teleop")
    cfg = env.unwrapped.cfg
    env.close()

    assert cfg.scene.robot.spawn.rigid_props.max_depenetration_velocity == pytest.approx(5.0)
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is False
    assert cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count == 32
    assert cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count == 1

    expected = {
        "rotation": (55, 0.7),
        "pitch": (30, 0.8),
        "elbow": (25, 0.7),
        "wrist_pitch": (12, 0.5),
        "wrist_roll": (7, 0.5),
        "gripper": (4, 0.3),
    }
    for name, (stiffness, damping) in expected.items():
        actuator = cfg.scene.robot.actuators[name]
        assert actuator.effort_limit_sim == 30, name
        assert actuator.stiffness == pytest.approx(stiffness), name
        assert actuator.damping == pytest.approx(damping), name


def test_so101_rl_uses_compliant_lior_style_actuators():
    """RL config adopts Lior's compliant gains with three pinned divergences.

    See SO_ARM101_CFG in so_arm101.py for the rationale. Divergences from
    SO_ARM101_TELEOP_CFG: activate_contact_sensors=True (grasped_reward),
    velocity_limit_sim capped on every actuator (prevents the high-effort
    actuator from slamming toward policy targets), and RL gripper k=15
    (k=4 closes too slowly to pin a moving cube during exploration).
    """
    from openso101.robots.so101.so_arm101 import SO101_RL_VELOCITY_LIMIT

    env = gym.make("OpenSO101-PickPlace-v0")
    cfg = env.unwrapped.cfg
    env.close()

    assert cfg.scene.robot.spawn.activate_contact_sensors is True
    assert cfg.scene.robot.spawn.rigid_props.max_depenetration_velocity == pytest.approx(5.0)
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is False
    assert cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count == 32
    assert cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count == 1

    expected = {
        "rotation": (55, 0.7),
        "pitch": (30, 0.8),
        "elbow": (25, 0.7),
        "wrist_pitch": (12, 0.5),
        "wrist_roll": (7, 0.5),
        "gripper": (15, 0.5),
    }
    for name, (stiffness, damping) in expected.items():
        actuator = cfg.scene.robot.actuators[name]
        assert actuator.effort_limit_sim == 30, name
        assert actuator.stiffness == pytest.approx(stiffness), name
        assert actuator.damping == pytest.approx(damping), name
        assert actuator.velocity_limit_sim == pytest.approx(SO101_RL_VELOCITY_LIMIT), name


def test_so101_cameras_use_upstream_model_wrist_mount_and_overhead_camera():
    env = gym.make("OpenSO101-PickPlace-v0", cameras=True)
    cfg = env.unwrapped.cfg
    env.close()

    assert cfg.scene.wrist_camera.prim_path == "{ENV_REGEX_NS}/Robot/gripper/gripper_cam"
    assert cfg.scene.overhead_camera.prim_path == "{ENV_REGEX_NS}/overhead_cam"
    assert cfg.scene.wrist_camera.offset.pos == pytest.approx((-0.005, 0.06, -0.062))
    assert cfg.scene.overhead_camera.offset.pos == pytest.approx((0.34412, 0.41122, 0.4191))
