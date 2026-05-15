# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Best-effort Isaac viewport helpers for teleop cameras."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .lerobot_recorder import REQUIRED_CAMERA_NAMES, get_scene_entity, has_scene_entity


@dataclass(frozen=True)
class TeleopViewportSpec:
    name: str
    camera_path: str | None


@dataclass(frozen=True)
class TeleopViewportDockSpec:
    viewport: TeleopViewportSpec
    dock_target: str | None
    dock_position: str | None
    dock_ratio: float


@dataclass(frozen=True)
class TeleopViewportTileSpec:
    viewport: TeleopViewportSpec
    position_x: int
    position_y: int
    width: int
    height: int


def camera_prim_path(sensor: Any) -> str:
    """Return the resolved USD camera prim path for an Isaac camera sensor."""

    prim_paths = getattr(sensor, "prim_paths", None)
    if prim_paths:
        return str(prim_paths[0])

    view = getattr(sensor, "_view", None)
    view_prim_paths = getattr(view, "prim_paths", None)
    if view_prim_paths:
        return str(view_prim_paths[0])

    prim_path = getattr(sensor, "prim_path", None)
    if prim_path:
        return str(prim_path)

    return str(sensor.cfg.prim_path)


def teleop_viewport_specs(scene: Mapping[str, Any]) -> list[TeleopViewportSpec]:
    """Return the three teleop viewport specs: default, wrist, overhead."""

    specs = [TeleopViewportSpec("Default Camera", None)]
    for camera_name in REQUIRED_CAMERA_NAMES:
        if has_scene_entity(scene, camera_name):
            specs.append(TeleopViewportSpec(camera_name.replace("_", " ").title(), camera_prim_path(get_scene_entity(scene, camera_name))))
    return specs


def teleop_viewport_dock_specs(scene: Mapping[str, Any]) -> list[TeleopViewportDockSpec]:
    """Return a one-app-window layout for the default, wrist, and overhead viewports."""

    specs = teleop_viewport_specs(scene)
    dock_specs: list[TeleopViewportDockSpec] = []
    for index, spec in enumerate(specs):
        if index == 0:
            dock_specs.append(TeleopViewportDockSpec(spec, dock_target=None, dock_position=None, dock_ratio=1.0))
        elif index == 1:
            dock_specs.append(TeleopViewportDockSpec(spec, dock_target=specs[0].name, dock_position="RIGHT", dock_ratio=0.35))
        else:
            dock_specs.append(
                TeleopViewportDockSpec(spec, dock_target=specs[index - 1].name, dock_position="BOTTOM", dock_ratio=0.5)
            )
    return dock_specs


def teleop_viewport_tile_specs(
    scene: Mapping[str, Any],
    main_width: int = 1600,
    main_height: int = 900,
) -> list[TeleopViewportTileSpec]:
    """Return deterministic teleop viewport tiles: default left, wrist/overhead stacked right."""

    specs = teleop_viewport_specs(scene)
    width = max(int(main_width), 960)
    height = max(int(main_height) - 44, 540)
    top = 36
    left_width = max(int(width * 0.64), 640)
    right_width = max(width - left_width, 360)
    camera_height = max(height // 2, 260)

    tiles: list[TeleopViewportTileSpec] = []
    for index, spec in enumerate(specs):
        if index == 0:
            tiles.append(TeleopViewportTileSpec(spec, position_x=0, position_y=top, width=left_width, height=height))
        elif index == 1:
            tiles.append(
                TeleopViewportTileSpec(
                    spec,
                    position_x=left_width,
                    position_y=top,
                    width=right_width,
                    height=camera_height,
                )
            )
        else:
            tiles.append(
                TeleopViewportTileSpec(
                    spec,
                    position_x=left_width,
                    position_y=top + camera_height,
                    width=right_width,
                    height=max(height - camera_height, 260),
                )
            )
    return tiles


def _hide_editor_windows(ui) -> None:
    """Give teleop viewports the main Isaac window instead of the default editor panels."""

    for window_name in (
        "Content",
        "Console",
        "Stage",
        "Layer",
        "Render Settings",
        "Property",
        "Semantics",
        "Semantics Schema Editor",
        "IsaacLab",
    ):
        try:
            window = ui.Workspace.get_window(window_name)
            if window is not None:
                window.visible = False
        except Exception:
            pass


def _hide_existing_viewport_windows() -> None:
    try:
        from omni.kit.viewport.window import get_viewport_window_instances

        for window in get_viewport_window_instances(None):
            try:
                if not str(window.title).startswith("Teleop "):
                    window.visible = False
            except Exception:
                pass
    except Exception:
        pass


def _disable_viewport_overlays() -> None:
    """Hide viewport gizmos so teleop camera panes show only the camera feed."""

    try:
        import carb.settings

        settings = carb.settings.get_settings()
        settings.set("/exts/omni.kit.hydra_texture/gizmos/enabled", False)
    except Exception:
        pass


def _focus_window(window) -> None:
    try:
        window.visible = True
    except Exception:
        pass
    try:
        window.bring_to_front()
    except Exception:
        pass
    try:
        window.focus()
    except Exception:
        pass


def _set_viewport_camera(viewport_api, camera_path: str | None) -> bool:
    if camera_path is None:
        return True
    try:
        from omni.kit.viewport.actions import set_camera

        if set_camera(camera_path, viewport_api):
            return True
    except Exception:
        pass
    try:
        from pxr import Sdf

        viewport_api.camera_path = Sdf.Path(camera_path)
        return str(viewport_api.camera_path) == camera_path
    except Exception:
        pass
    try:
        viewport_api.camera_path = camera_path
        return str(viewport_api.camera_path) == camera_path
    except Exception:
        return False


def open_teleop_viewports(scene: Mapping[str, Any]) -> None:
    """Open default, wrist, and overhead teleop viewports as deterministic panes in one Isaac app window."""

    try:
        import omni.ui as ui
        from omni.kit.viewport.utility import create_viewport_window
    except Exception as exc:  # pragma: no cover - depends on Isaac UI runtime
        print(f"[WARN]: Camera viewport utility unavailable: {exc}")
        return

    _hide_editor_windows(ui)
    _hide_existing_viewport_windows()
    _disable_viewport_overlays()

    for camera_name in REQUIRED_CAMERA_NAMES:
        if not has_scene_entity(scene, camera_name):
            print(f"[WARN]: Cannot open {camera_name} viewport; sensor is missing.")

    try:
        main_width = int(ui.Workspace.get_main_window_width())
        main_height = int(ui.Workspace.get_main_window_height())
    except Exception:
        main_width, main_height = 1600, 900

    windows: dict[str, Any] = {}
    for tile in teleop_viewport_tile_specs(scene, main_width=main_width, main_height=main_height):
        spec = tile.viewport
        try:
            from pxr import Sdf

            camera_path = Sdf.Path(spec.camera_path) if spec.camera_path is not None else None
            window = create_viewport_window(
                f"Teleop {spec.name}",
                width=tile.width,
                height=tile.height,
                position_x=tile.position_x,
                position_y=tile.position_y,
                camera_path=camera_path,
            )
            if window is None:
                print(f"[WARN]: Failed to create {spec.name} viewport.")
                continue
            viewport_api = window.viewport_api
            try:
                window.setPosition(tile.position_x, tile.position_y)
            except Exception:
                pass
            if spec.camera_path is not None and not _set_viewport_camera(viewport_api, spec.camera_path):
                print(f"[WARN]: Could not bind {spec.name} viewport to camera prim {spec.camera_path}.")
            _disable_viewport_overlays()
            windows[spec.name] = window
            _focus_window(window)
            print(
                f"[INFO]: Opened {spec.name} viewport at {viewport_api.camera_path} "
                f"({tile.width}x{tile.height} at {tile.position_x},{tile.position_y})."
            )
        except Exception as exc:  # pragma: no cover - depends on Isaac UI runtime
            print(f"[WARN]: Failed to open {spec.name} viewport: {exc}")

    for spec_name in ("Overhead Camera", "Wrist Camera", "Default Camera"):
        if spec_name in windows:
            _focus_window(windows[spec_name])


def open_camera_viewports(scene: Mapping[str, Any]) -> None:
    """Backward-compatible alias for opening all teleop viewports."""

    open_teleop_viewports(scene)
