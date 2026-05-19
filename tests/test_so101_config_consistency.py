import pytest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_SRC = REPO_ROOT / "src" / "openso101" / "tasks"
SO101_CFG_SRC = REPO_ROOT / "src" / "openso101" / "robots" / "so101" / "so_arm101.py"
SO101_RL_DEFAULTS_SRC = TASK_SRC / "shared" / "rl_defaults.py"
SO101_PICK_PLACE_SRC = TASK_SRC / "pick_place" / "pick_place_env_cfg.py"


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


def test_so101_canonical_arm_actuators_use_lior_compliant_gains():
    """The canonical arm actuators match liorbenhorin/lerobot_so101_teleop.

    Switched from URDF-era high-stiffness (k=200, d=80 on rotation) to Lior's
    compliant low-stiffness gains because the high values made the gripper
    hammer rather than pinch. Compliant gains require effort_limit_sim=30 so
    the actuator has headroom to reach targets.

    velocity_limit_sim is set on RL (but not teleop) to cap the arm speed.
    Without it the compliant actuator slams toward policy-issued targets at
    ~25 rad/s; teleop's leader-driven targets are naturally bounded.
    """
    pytest.importorskip("isaaclab.sim")

    from openso101.robots.so101.so_arm101 import SO_ARM101_CFG, SO101_RL_VELOCITY_LIMIT

    expected = {
        "rotation":    dict(joint="Rotation",    stiffness=55, damping=0.7),
        "pitch":       dict(joint="Pitch",       stiffness=30, damping=0.8),
        "elbow":       dict(joint="Elbow",       stiffness=25, damping=0.7),
        "wrist_pitch": dict(joint="Wrist_Pitch", stiffness=12, damping=0.5),
        "wrist_roll":  dict(joint="Wrist_Roll", stiffness=7,  damping=0.5),
    }
    for name, want in expected.items():
        act = SO_ARM101_CFG.actuators[name]
        assert act.joint_names_expr == [want["joint"]], (name, act.joint_names_expr)
        assert act.effort_limit_sim == 30, name
        assert act.velocity_limit_sim == pytest.approx(SO101_RL_VELOCITY_LIMIT), name
        assert act.stiffness == pytest.approx(want["stiffness"]), name
        assert act.damping == pytest.approx(want["damping"]), name


def test_so101_canonical_gripper_actuator_uses_rl_tuned_gains():
    """RL gripper stiffness is bumped from Lior's k=4 to k=15.

    Lior's k=4 is great for teleop (human leads, gripper follows), but for
    RL the slow gripper closure means the policy can't reliably pin a moving
    cube during exploration; entropy collapses before the policy ever
    discovers "close gripper at the right moment". k=15 closes fast enough
    for RL while still being well below the URDF-era 60 that hammered the
    cube and bounced it out.
    """
    pytest.importorskip("isaaclab.sim")

    from openso101.robots.so101.so_arm101 import SO_ARM101_CFG, SO101_RL_VELOCITY_LIMIT

    act = SO_ARM101_CFG.actuators["gripper"]
    assert act.joint_names_expr == ["Jaw"]
    assert act.effort_limit_sim == 30
    assert act.velocity_limit_sim == pytest.approx(SO101_RL_VELOCITY_LIMIT)
    assert act.stiffness == pytest.approx(15)
    assert act.damping == pytest.approx(0.5)


def test_so101_canonical_articulation_props_match_lior():
    """The canonical articulation props match Lior's compliant config: solver
    32/1 and self-collisions disabled so the arm doesn't bind."""
    pytest.importorskip("isaaclab.sim")

    from openso101.robots.so101.so_arm101 import SO_ARM101_CFG

    props = SO_ARM101_CFG.spawn.articulation_props
    assert props.enabled_self_collisions is False
    assert props.solver_position_iteration_count == 32
    assert props.solver_velocity_iteration_count == 1
    assert props.fix_root_link is True
    assert SO_ARM101_CFG.soft_joint_pos_limit_factor == pytest.approx(0.9)


def test_so101_ppo_uses_log_noise_std_parameterization():
    """All rsl_rl PPO configs must use noise_std_type='log' so the policy's
    action-noise std cannot drift negative. Pinned after the 2026-05-14 lift
    training crash at iter 79."""
    text = SO101_RL_DEFAULTS_SRC.read_text()

    assert 'SO101_PPO_NOISE_STD_TYPE: str = "log"' in text


def test_so101_rl_and_teleop_share_lior_compliant_config():
    """RL and teleop both adopt Lior's compliant gains and solver settings.

    Two intentional divergences:
    - activate_contact_sensors: RL=True (pick_place's grasped_reward needs
      contact signals on /Robot/gripper and /Robot/jaw); teleop=False.
    - velocity_limit_sim: RL caps every joint at SO101_RL_VELOCITY_LIMIT;
      teleop leaves it unset (matches Lior verbatim). Without the cap, the
      RL policy can issue position deltas that whip the arm at ~25 rad/s
      because the compliant actuator has 30 N-m of effort headroom and only
      ~0.7 damping. Teleop doesn't need the cap because the leader-arm-
      driven targets are naturally bounded by human motion.

    Everything else (stiffness, damping, effort cap, solver, self-collisions)
    is identical so sim behavior stays consistent between IL and RL.
    """
    pytest.importorskip("isaaclab.sim")

    from openso101.robots.so101.so_arm101 import (
        SO_ARM101_CFG,
        SO_ARM101_TELEOP_CFG,
        SO101_RL_VELOCITY_LIMIT,
    )

    # Actuator parity on the arm joints: Lior's compliant gains on both,
    # effort=30. Velocity capped on RL only. Gripper diverges (see below).
    for name in ("rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll"):
        rl_act = SO_ARM101_CFG.actuators[name]
        tp_act = SO_ARM101_TELEOP_CFG.actuators[name]
        assert rl_act.effort_limit_sim == tp_act.effort_limit_sim == 30, name
        assert rl_act.stiffness == pytest.approx(tp_act.stiffness), name
        assert rl_act.damping == pytest.approx(tp_act.damping), name
        assert rl_act.velocity_limit_sim == pytest.approx(SO101_RL_VELOCITY_LIMIT), name
        assert tp_act.velocity_limit_sim is None, name

    # Gripper diverges: teleop keeps Lior's k=4 (works for human-led grip);
    # RL bumps to k=15 (fast enough to pin a moving cube during exploration).
    rl_grip = SO_ARM101_CFG.actuators["gripper"]
    tp_grip = SO_ARM101_TELEOP_CFG.actuators["gripper"]
    assert rl_grip.effort_limit_sim == tp_grip.effort_limit_sim == 30
    assert rl_grip.stiffness == pytest.approx(15)
    assert tp_grip.stiffness == pytest.approx(4)
    assert rl_grip.damping == pytest.approx(0.5)
    assert tp_grip.damping == pytest.approx(0.3)

    # Articulation parity, except the contact-sensor flag.
    for cfg in (SO_ARM101_CFG, SO_ARM101_TELEOP_CFG):
        assert cfg.spawn.articulation_props.enabled_self_collisions is False
        assert cfg.spawn.articulation_props.solver_position_iteration_count == 32
        assert cfg.spawn.articulation_props.solver_velocity_iteration_count == 1
    assert SO_ARM101_CFG.spawn.activate_contact_sensors is True
    assert SO_ARM101_TELEOP_CFG.spawn.activate_contact_sensors is False

    # The two cfgs must be different ArticulationCfg objects (no shared identity).
    assert SO_ARM101_CFG is not SO_ARM101_TELEOP_CFG
    assert SO_ARM101_CFG.actuators is not SO_ARM101_TELEOP_CFG.actuators
