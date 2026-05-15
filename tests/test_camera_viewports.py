import pytest

pytest.skip(
    "ported from safe_sim2real under OpenSO-101 skeleton mode; "
    "awaits implementation port. See "
    "/data/safe_sim2real/tests/test_camera_viewports.py for the legacy assertions.",
    allow_module_level=True,
)

from openso101.teleop.camera_viewports import (
    camera_prim_path,
    teleop_viewport_dock_specs,
    teleop_viewport_specs,
    teleop_viewport_tile_specs,
)


class FakeSensor:
    def __init__(self, prim_path=None, prim_paths=None, view_prim_paths=None):
        self.prim_path = prim_path
        self.prim_paths = prim_paths
        self._view = type("View", (), {"prim_paths": view_prim_paths})() if view_prim_paths is not None else None
        self.cfg = type("Cfg", (), {"prim_path": "{ENV_REGEX_NS}/fallback"})()


def test_camera_prim_path_prefers_runtime_prim_path():
    assert camera_prim_path(FakeSensor(prim_path="/World/envs/env_0/wrist_cam")) == "/World/envs/env_0/wrist_cam"


def test_camera_prim_path_uses_first_runtime_prim_paths_entry():
    assert camera_prim_path(FakeSensor(prim_paths=["/World/envs/env_0/overhead_cam"])) == "/World/envs/env_0/overhead_cam"


def test_camera_prim_path_prefers_resolved_prim_paths_over_config_pattern():
    assert (
        camera_prim_path(
            FakeSensor(
                prim_path="{ENV_REGEX_NS}/overhead_cam",
                prim_paths=["/World/envs/env_0/overhead_cam"],
            )
        )
        == "/World/envs/env_0/overhead_cam"
    )


def test_camera_prim_path_uses_isaac_lab_tiled_camera_view_paths():
    assert (
        camera_prim_path(
            FakeSensor(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_cam",
                view_prim_paths=["/World/envs/env_0/Robot/gripper_link/wrist_cam"],
            )
        )
        == "/World/envs/env_0/Robot/gripper_link/wrist_cam"
    )


def test_teleop_viewport_specs_include_default_wrist_and_overhead_views():
    scene = {
        "wrist_camera": FakeSensor(prim_path="/World/envs/env_0/wrist_cam"),
        "overhead_camera": FakeSensor(prim_path="/World/envs/env_0/overhead_cam"),
    }

    specs = teleop_viewport_specs(scene)

    assert [spec.name for spec in specs] == ["Default Camera", "Wrist Camera", "Overhead Camera"]
    assert specs[0].camera_path is None
    assert specs[1].camera_path == "/World/envs/env_0/wrist_cam"
    assert specs[2].camera_path == "/World/envs/env_0/overhead_cam"


def test_teleop_viewports_are_arranged_as_docked_panes_in_one_app_window():
    scene = {
        "wrist_camera": FakeSensor(prim_path="/World/envs/env_0/wrist_cam"),
        "overhead_camera": FakeSensor(prim_path="/World/envs/env_0/overhead_cam"),
    }

    layout = teleop_viewport_dock_specs(scene)

    assert [item.viewport.name for item in layout] == ["Default Camera", "Wrist Camera", "Overhead Camera"]
    assert layout[0].dock_target is None
    assert layout[1].dock_target == "Default Camera"
    assert layout[1].dock_position == "RIGHT"
    assert layout[1].dock_ratio == 0.35
    assert layout[2].dock_target == "Wrist Camera"
    assert layout[2].dock_position == "BOTTOM"
    assert layout[2].dock_ratio == 0.5


def test_teleop_viewports_are_tiled_as_default_left_and_camera_column_right():
    scene = {
        "wrist_camera": FakeSensor(prim_path="/World/envs/env_0/wrist_cam"),
        "overhead_camera": FakeSensor(prim_path="/World/envs/env_0/overhead_cam"),
    }

    tiles = teleop_viewport_tile_specs(scene, main_width=1600, main_height=900)

    assert [tile.viewport.name for tile in tiles] == ["Default Camera", "Wrist Camera", "Overhead Camera"]
    assert tiles[0].position_x == 0
    assert tiles[0].position_y == tiles[1].position_y
    assert tiles[1].position_x == tiles[0].width
    assert tiles[2].position_x == tiles[1].position_x
    assert tiles[2].position_y == tiles[1].position_y + tiles[1].height
    assert tiles[1].width == tiles[2].width
