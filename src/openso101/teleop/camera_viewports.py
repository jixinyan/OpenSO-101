# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Best-effort Isaac viewport helpers for teleop cameras.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/teleop/camera_viewports.py` lands. The
legacy implementation is under active revision (motor-unit remap,
streaming HDF5 recording); this skeleton freezes the public surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TeleopViewportSpec:
    """Name and optional USD camera prim path for a single teleop viewport."""

    name: str
    camera_path: str | None


@dataclass(frozen=True)
class TeleopViewportDockSpec:
    """Docking parameters for a teleop viewport within Isaac's main window."""

    viewport: TeleopViewportSpec
    dock_target: str | None
    dock_position: str | None
    dock_ratio: float


@dataclass(frozen=True)
class TeleopViewportTileSpec:
    """Absolute pixel-position tile spec for a teleop viewport."""

    viewport: TeleopViewportSpec
    position_x: int
    position_y: int
    width: int
    height: int


def camera_prim_path(sensor: Any) -> str:
    """Return the resolved USD camera prim path for an Isaac camera sensor."""
    raise NotImplementedError(
        "camera_prim_path not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/camera_viewports.py"
    )


def teleop_viewport_specs(scene: Mapping[str, Any]) -> list[TeleopViewportSpec]:
    """Return the three teleop viewport specs: default, wrist, overhead."""
    raise NotImplementedError(
        "teleop_viewport_specs not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/camera_viewports.py"
    )


def teleop_viewport_dock_specs(scene: Mapping[str, Any]) -> list[TeleopViewportDockSpec]:
    """Return a one-app-window layout for the default, wrist, and overhead viewports."""
    raise NotImplementedError(
        "teleop_viewport_dock_specs not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/camera_viewports.py"
    )


def teleop_viewport_tile_specs(
    scene: Mapping[str, Any],
    main_width: int = 1600,
    main_height: int = 900,
) -> list[TeleopViewportTileSpec]:
    """Return deterministic teleop viewport tiles: default left, wrist/overhead stacked right."""
    raise NotImplementedError(
        "teleop_viewport_tile_specs not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/camera_viewports.py"
    )


def open_teleop_viewports(scene: Mapping[str, Any]) -> None:
    """Open default, wrist, and overhead teleop viewports as deterministic panes in one Isaac app window."""
    raise NotImplementedError(
        "open_teleop_viewports not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/camera_viewports.py"
    )


def open_camera_viewports(scene: Mapping[str, Any]) -> None:
    """Backward-compatible alias for opening all teleop viewports."""
    raise NotImplementedError(
        "open_camera_viewports not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/camera_viewports.py"
    )


__all__ = [
    "TeleopViewportSpec",
    "TeleopViewportDockSpec",
    "TeleopViewportTileSpec",
    "camera_prim_path",
    "teleop_viewport_specs",
    "teleop_viewport_dock_specs",
    "teleop_viewport_tile_specs",
    "open_teleop_viewports",
    "open_camera_viewports",
]
