import pytest

pytest.importorskip("isaaclab_tasks")

import gymnasium as gym
import isaaclab.sim as sim_utils

import openso101.robots as robots
import openso101.tasks.pick_place  # noqa: F401
from openso101.robots.so101.so_arm101 import spawn_so101_usd_with_safe_collisions
from openso101.robots.so101.so_arm101 import SO101_USD_TABLETOP_ROOT_Z


def test_so101_training_scene_uses_canonical_usd_robot_backend():
    env = gym.make("OpenSO101-PickPlace-v0")
    cfg = env.unwrapped.cfg
    env.close()

    assert cfg.scene.robot.spawn.usd_path.endswith("SO-ARM101-USD.usd")
    assert cfg.scene.robot.spawn.func is spawn_so101_usd_with_safe_collisions
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


def test_so101_pick_place_curriculum_heights_follow_lowered_usd_root():
    env = gym.make("OpenSO101-PickPlace-v0")
    cfg = env.unwrapped.cfg
    env.close()
    root_drop = -SO101_USD_TABLETOP_ROOT_Z

    assert cfg.commands.object_pose.lift_height == pytest.approx(0.10 + root_drop)
    assert cfg.commands.object_pose.carry_height == pytest.approx(0.15 + root_drop)
    assert cfg.commands.object_pose.place_goal == pytest.approx((0.20, 0.18, 0.02 + root_drop))
    assert cfg.rewards.stage_0_lift_goal.params["minimal_height"] == pytest.approx(0.04 + root_drop)
    assert cfg.rewards.stage_1_carry_goal.params["minimal_height"] == pytest.approx(0.04 + root_drop)


def test_so101_robot_public_exports_only_canonical_config():
    assert "SO_ARM101_CFG" in robots.__all__
    assert "SO_ARM101_UPSTREAM_USD_CFG" not in robots.__all__
    assert not hasattr(robots, "SO_ARM101_UPSTREAM_USD_CFG")


def test_pick_place_registry_has_no_separate_usd_ab_task_ids():
    assert gym.spec("OpenSO101-PickPlace-v0").kwargs["env_cfg_entry_point"].endswith(":OpenSO101PickPlaceEnvCfg")
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
    assert cfg.scene.robot.spawn.func is spawn_so101_usd_with_safe_collisions
    assert cfg.actions.joint_positions.joint_names == ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    assert cfg.scene.fixed_fingertip_pad is None
    assert cfg.scene.moving_fingertip_pad is None
    assert cfg.rewards is None
    assert cfg.terminations is None
    assert cfg.sim.dt == pytest.approx(1 / 120)
    assert cfg.sim.render_interval == cfg.decimation


def test_so101_teleop_uses_soft_contact_robot_response():
    """Teleop config has its own actuator gains, independent of the RL canonical.

    Effort/velocity caps are tuned for tracking a human leader arm; self-
    collisions are disabled so the follower doesn't bind when the user drives
    the leader into awkward configurations.
    """
    env = gym.make("OpenSO101-PickPlace-v0", action_mode="teleop")
    cfg = env.unwrapped.cfg
    env.close()

    assert cfg.scene.robot.spawn.rigid_props.max_depenetration_velocity == pytest.approx(1.0)
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is False
    assert cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count == 32
    assert cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count == 4

    for name in ("rotation", "pitch", "elbow"):
        actuator = cfg.scene.robot.actuators[name]
        assert actuator.effort_limit_sim == pytest.approx(8.0), name
        assert actuator.velocity_limit_sim == pytest.approx(3.5), name
    for name in ("wrist_pitch", "wrist_roll"):
        actuator = cfg.scene.robot.actuators[name]
        assert actuator.effort_limit_sim == pytest.approx(4.0), name
        assert actuator.velocity_limit_sim == pytest.approx(2.0), name
    assert cfg.scene.robot.actuators["gripper"].effort_limit_sim == pytest.approx(4.0)
    assert cfg.scene.robot.actuators["gripper"].velocity_limit_sim == pytest.approx(2.0)


def test_so101_cameras_use_upstream_model_wrist_mount_and_overhead_camera():
    env = gym.make("OpenSO101-PickPlace-v0", cameras=True)
    cfg = env.unwrapped.cfg
    env.close()

    assert cfg.scene.wrist_camera.prim_path == "{ENV_REGEX_NS}/Robot/gripper/gripper_cam"
    assert cfg.scene.overhead_camera.prim_path == "{ENV_REGEX_NS}/overhead_cam"
    assert cfg.scene.wrist_camera.offset.pos == pytest.approx((-0.005, 0.06, -0.062))
    assert cfg.scene.overhead_camera.offset.pos == pytest.approx((0.34412, 0.41122, 0.4191))
