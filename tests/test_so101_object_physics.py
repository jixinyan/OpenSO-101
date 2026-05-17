import pytest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_SRC = REPO_ROOT / "src" / "openso101" / "tasks"
JOINT_POS_CFGS = sorted(TASK_SRC.rglob("joint_pos_env_cfg.py"))


def test_so101_manipulation_tasks_use_shared_cube_factory():
    offenders = []
    for path in JOINT_POS_CFGS:
        text = path.read_text()
        if "RigidObjectCfg(" in text and "so101_cube_object_cfg" not in text:
            offenders.append(path.relative_to(REPO_ROOT))

    assert offenders == []


def test_so101_task_configs_do_not_use_old_heavy_or_usd_cube_physics():
    forbidden = (
        "MassPropertiesCfg(mass=0.1)",
        "DexCube",
        "dex_cube_instanceable.usd",
        "_configure_teleop_cube_response",
    )
    offenders = []
    for path in TASK_SRC.rglob("*.py"):
        text = path.read_text()
        if any(pattern in text for pattern in forbidden):
            offenders.append(path.relative_to(REPO_ROOT))

    assert offenders == []


def test_so101_object_reset_events_do_not_assume_specific_body_name():
    offenders = []
    for path in TASK_SRC.rglob("*.py"):
        text = path.read_text()
        if 'SceneEntityCfg("object", body_names="Object")' in text:
            offenders.append(path.relative_to(REPO_ROOT))

    assert offenders == []


def test_prebuilt_block_probe_path_is_not_exposed_as_supported_teleop_physics():
    # In OpenSO-101 the teleop_agent.py body lives inline in cli/il.py.
    teleop_agent_path = REPO_ROOT / "src" / "openso101" / "cli" / "il.py"
    object_cfg_path = TASK_SRC / "shared" / "objects.py"

    teleop_agent = teleop_agent_path.read_text()
    object_cfg = object_cfg_path.read_text()

    forbidden = ("--object-asset", "blue_block", "red_block", "green_block", "Props/Blocks")
    offenders = [pattern for pattern in forbidden if pattern in teleop_agent or pattern in object_cfg]

    assert offenders == []


def test_so101_robot_module_does_not_reintroduce_custom_collision_spawn():
    """Pin the absence of the custom collision spawn func and friction constants.

    Earlier rewrites tried to "improve" the upstream USD's authored colliders
    by walking the gripper subtree and applying standalone CollisionAPI plus
    bound friction materials. That conflicted with the USD's
    PhysxMeshMergeCollisionAPI on the parent bodies and silently disabled
    gripper collision. The aggressive Lior-style port deletes the function
    entirely; this test ensures no future regression re-adds it.
    """
    robot_cfg = (REPO_ROOT / "src" / "openso101" / "robots" / "so101" / "so_arm101.py").read_text()

    assert "spawn_so101_usd_with_safe_collisions" not in robot_cfg
    assert "SO101_GRIPPER_CONTACT_STATIC_FRICTION" not in robot_cfg
    assert "SO101_GRIPPER_CONTACT_DYNAMIC_FRICTION" not in robot_cfg
    assert "SO101_GRIPPER_CONTACT_OFFSET" not in robot_cfg
    assert "MaterialBindingAPI" not in robot_cfg
