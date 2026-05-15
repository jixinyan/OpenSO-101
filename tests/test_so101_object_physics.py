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
    teleop_agent_path = REPO_ROOT / "src" / "openso101" / "scripts" / "lerobot" / "teleop_agent.py"
    object_cfg_path = TASK_SRC / "shared" / "objects.py"
    if not teleop_agent_path.exists() or not object_cfg_path.exists():
        pytest.skip("requires concrete teleop_agent.py and shared/objects.py (skeleton mode)")

    teleop_agent = teleop_agent_path.read_text()
    object_cfg = object_cfg_path.read_text()

    forbidden = ("--object-asset", "blue_block", "red_block", "green_block", "Props/Blocks")
    offenders = [pattern for pattern in forbidden if pattern in teleop_agent or pattern in object_cfg]

    assert offenders == []


@pytest.mark.skip(reason="skeleton mode: so_arm101.py body (USD spawn, friction binding) not yet ported")
def test_so101_gripper_contact_collisions_get_contact_material_and_offsets():
    robot_cfg = (REPO_ROOT / "src" / "openso101" / "robots" / "so101" / "so_arm101.py").read_text()

    assert "SO101_GRIPPER_CONTACT_STATIC_FRICTION" in robot_cfg
    assert "SO101_GRIPPER_CONTACT_DYNAMIC_FRICTION" in robot_cfg
    assert "SO101_GRIPPER_CONTACT_OFFSET" in robot_cfg
    assert "MaterialBindingAPI" in robot_cfg
    assert 'approx_attr.Set("convexHull")' in robot_cfg
