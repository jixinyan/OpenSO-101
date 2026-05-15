import pytest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_SRC = REPO_ROOT / "src" / "openso101" / "tasks"
SO101_CFG_SRC = REPO_ROOT / "src" / "openso101" / "robots" / "so101" / "so_arm101.py"
SO101_RL_DEFAULTS_SRC = TASK_SRC / "shared" / "rl_defaults.py"
SO101_PICK_PLACE_SRC = TASK_SRC / "pick_place" / "pick_place_env_cfg.py"
SO101_PICK_PLACE_MDP_INIT_SRC = TASK_SRC / "pick_place" / "mdp" / "__init__.py"


def _task_python_files() -> list[Path]:
    return sorted(TASK_SRC.rglob("*.py"))


def test_task_code_does_not_depend_on_teleop_mapping():
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in _task_python_files()
        if "openso101.teleop" in path.read_text()
    ]

    assert offenders == []


def test_so101_joint_action_configs_use_robot_constants():
    literal_arm_action = 'joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]'
    literal_gripper_action = 'joint_names=["Jaw"]'
    offenders = []
    for path in _task_python_files():
        text = path.read_text()
        if literal_arm_action in text or literal_gripper_action in text:
            offenders.append(path.relative_to(REPO_ROOT))

    assert offenders == []


def test_so101_canonical_robot_is_fixed_base():
    text = SO101_CFG_SRC.read_text()

    assert "fix_root_link=True" in text
    assert "pos=(0.0, 0.0, SO101_USD_TABLETOP_ROOT_Z)" in text
    assert "SO101_GRIPPER_OPEN_POS" in text


def test_so101_tasks_do_not_override_canonical_robot_reset_pose():
    offenders = []
    forbidden = (
        ".scene.robot.init_state.pos",
        ".scene.robot.init_state.joint_pos[SO101_GRIPPER_JOINT_NAME]",
    )
    for path in _task_python_files():
        text = path.read_text()
        if any(pattern in text for pattern in forbidden):
            offenders.append(path.relative_to(REPO_ROOT))

    assert offenders == []


def test_so101_rl_defaults_are_shared_by_standard_ppo_agents():
    text = SO101_RL_DEFAULTS_SRC.read_text()
    # Two-tanh reach: coarse kernel for long-range + fine for grasp approach
    assert "SO101_REACH_REWARD_COARSE_STD = 0.20" in text
    assert "SO101_REACH_REWARD_STD = 0.05" in text
    assert "SO101_PPO_NUM_STEPS_PER_ENV = 96" in text

    offenders = []
    for path in sorted(TASK_SRC.rglob("agents/rsl_rl_ppo_cfg.py")):
        text = path.read_text()
        required = (
            "SO101_PPO_NUM_STEPS_PER_ENV",
            "SO101_PPO_INIT_NOISE_STD",
            "SO101_PPO_ENTROPY_COEF",
            "SO101_PPO_GAMMA",
            "empirical_normalization = True",
        )
        if not all(pattern in text for pattern in required):
            offenders.append(path.relative_to(REPO_ROOT))

    assert offenders == []


def test_so101_tasks_do_not_reintroduce_legacy_sparse_reward_defaults():
    offenders = []
    forbidden = (
        '"near_threshold": 0.06',
        '"grasp_distance_threshold": 0.07',
        "joint_pos_delta_l2, weight=-0.1",
        'params={"term_name": "action_rate", "weight": -1e-1',
        'params={"term_name": "joint_vel", "weight": -1e-1',
        'func=mdp.object_ee_distance, params={"std": 0.05}',
    )
    for path in _task_python_files():
        if path == SO101_RL_DEFAULTS_SRC:
            continue
        text = path.read_text()
        if any(pattern in text for pattern in forbidden):
            offenders.append(path.relative_to(REPO_ROOT))

    assert offenders == []


def test_so101_pick_place_calibrates_canonical_usd_tabletop_pose():
    """The pick-place env must place the SO101 base bottom on the table top.

    Concretely: cfg.scene.robot.init_state.pos.z must equal the helper-
    derived tabletop_root_z value. This asserts against the helper rather
    than a literal so future USD changes propagate automatically.
    """
    pytest.importorskip("isaaclab_tasks")

    from openso101.robots.so101._usd_bounds import tabletop_root_z
    from openso101.robots.so101.so_arm101 import so101_usd_path
    from openso101.tasks.pick_place.pick_place_env_cfg import PickPlaceEnvCfg

    cfg = PickPlaceEnvCfg()
    expected_z = tabletop_root_z(so101_usd_path())

    assert cfg.scene.robot.init_state.pos[2] == pytest.approx(expected_z, abs=1e-6)
    assert cfg.scene.robot.init_state.joint_pos["Jaw"] == pytest.approx(1.745, abs=1e-3)
    assert cfg.scene.ee_frame.target_frames[0].offset.pos == pytest.approx((0.01, 0.0, -0.09))


def test_pick_place_mdp_exports_close_gripper_proxy_reward():
    text = SO101_PICK_PLACE_MDP_INIT_SRC.read_text()

    assert "close_gripper_near_object" in text


def test_so101_canonical_arm_actuators_use_urdf_era_gains():
    """The canonical arm actuators must match the proven URDF-era gains.

    These values trained atomic_lift to 999 iters on 2026-05-09. Commit 5d01353
    swapped the canonical spawn from URDF to USD but kept the weak hand-tuned
    USD values; this test pins the restored URDF-era values.
    """
    pytest.importorskip("isaaclab.sim")

    from openso101.robots.so101.so_arm101 import SO_ARM101_CFG

    expected = {
        "rotation":    dict(joint="Rotation",    stiffness=200.0, damping=80.0),
        "pitch":       dict(joint="Pitch",       stiffness=170.0, damping=65.0),
        "elbow":       dict(joint="Elbow",       stiffness=120.0, damping=45.0),
        "wrist_pitch": dict(joint="Wrist_Pitch", stiffness=80.0,  damping=30.0),
        "wrist_roll":  dict(joint="Wrist_Roll", stiffness=50.0,  damping=20.0),
    }
    for name, want in expected.items():
        act = SO_ARM101_CFG.actuators[name]
        assert act.joint_names_expr == [want["joint"]], (name, act.joint_names_expr)
        assert act.effort_limit_sim == pytest.approx(1.9), name
        assert act.velocity_limit_sim == pytest.approx(1.5), name
        assert act.stiffness == pytest.approx(want["stiffness"]), name
        assert act.damping == pytest.approx(want["damping"]), name


def test_so101_canonical_gripper_actuator_uses_urdf_era_gains():
    """The canonical gripper must have the URDF-era gains so it can actually close."""
    pytest.importorskip("isaaclab.sim")

    from openso101.robots.so101.so_arm101 import SO_ARM101_CFG

    act = SO_ARM101_CFG.actuators["gripper"]
    assert act.joint_names_expr == ["Jaw"]
    assert act.effort_limit_sim == pytest.approx(2.5)
    assert act.velocity_limit_sim == pytest.approx(1.5)
    assert act.stiffness == pytest.approx(60.0)
    assert act.damping == pytest.approx(20.0)


def test_so101_canonical_articulation_props_match_urdf_era():
    """The canonical articulation properties must match the URDF-era settings."""
    pytest.importorskip("isaaclab.sim")

    from openso101.robots.so101.so_arm101 import SO_ARM101_CFG

    props = SO_ARM101_CFG.spawn.articulation_props
    assert props.enabled_self_collisions is True
    assert props.solver_position_iteration_count == 8
    assert props.solver_velocity_iteration_count == 0
    assert props.fix_root_link is True
    assert SO_ARM101_CFG.soft_joint_pos_limit_factor == pytest.approx(0.9)


def test_so101_ppo_uses_log_noise_std_parameterization():
    """All rsl_rl PPO configs must use noise_std_type='log' so the policy's
    action-noise std cannot drift negative. Pinned after the 2026-05-14 lift
    training crash at iter 79."""
    text = SO101_RL_DEFAULTS_SRC.read_text()

    assert 'SO101_PPO_NOISE_STD_TYPE: str = "log"' in text


def test_so101_rl_canonical_independent_of_teleop_cfg():
    """The RL canonical must stay at URDF-era gains after introducing the teleop cfg.

    Pins that adding SO_ARM101_TELEOP_CFG did not accidentally drift the RL
    canonical's actuator values or articulation properties.
    """
    pytest.importorskip("isaaclab.sim")

    from openso101.robots.so101.so_arm101 import (
        SO_ARM101_CFG,
        SO_ARM101_TELEOP_CFG,
    )

    # RL canonical: 1.9 / 1.5 effort/velocity on the arm.
    rl_rotation = SO_ARM101_CFG.actuators["rotation"]
    assert rl_rotation.effort_limit_sim == pytest.approx(1.9)
    assert rl_rotation.velocity_limit_sim == pytest.approx(1.5)
    # Self-collisions on, solver iters 8/0.
    assert SO_ARM101_CFG.spawn.articulation_props.enabled_self_collisions is True
    assert SO_ARM101_CFG.spawn.articulation_props.solver_position_iteration_count == 8
    assert SO_ARM101_CFG.spawn.articulation_props.solver_velocity_iteration_count == 0

    # Teleop cfg: 8.0 / 3.5 effort/velocity on the arm.
    teleop_rotation = SO_ARM101_TELEOP_CFG.actuators["rotation"]
    assert teleop_rotation.effort_limit_sim == pytest.approx(8.0)
    assert teleop_rotation.velocity_limit_sim == pytest.approx(3.5)
    # Self-collisions off, solver iters 32/4.
    assert SO_ARM101_TELEOP_CFG.spawn.articulation_props.enabled_self_collisions is False
    assert SO_ARM101_TELEOP_CFG.spawn.articulation_props.solver_position_iteration_count == 32
    assert SO_ARM101_TELEOP_CFG.spawn.articulation_props.solver_velocity_iteration_count == 4

    # The two cfgs must be different ArticulationCfg objects (no shared identity).
    assert SO_ARM101_CFG is not SO_ARM101_TELEOP_CFG
    assert SO_ARM101_CFG.actuators is not SO_ARM101_TELEOP_CFG.actuators
