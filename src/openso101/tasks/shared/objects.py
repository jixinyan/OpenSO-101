# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Shared SO-101 manipulation object physics defaults (cube).

SKELETON: real constant values will be ported from
`/data/safe_sim2real/src/safe_sim2real/tasks/so101_object_cfg.py` once
the source is finalized.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import RigidObjectCfg
    import isaaclab.sim as sim_utils
    from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg


SO101_CUBE_SIZE: float = 0.0
SO101_CUBE_MASS: float = 0.0
SO101_CUBE_CONTACT_OFFSET: float = 0.0
SO101_CUBE_REST_OFFSET: float = 0.0
SO101_CUBE_STATIC_FRICTION: float = 0.0
SO101_CUBE_DYNAMIC_FRICTION: float = 0.0
SO101_CUBE_RESTITUTION: float = 0.0


def so101_cube_rigid_props(
    *,
    solver_position_iteration_count: int = 16,
    solver_velocity_iteration_count: int = 1,
    max_depenetration_velocity: float = 5.0,
) -> "RigidBodyPropertiesCfg":
    """Return rigid-body solver settings shared by SO-101 cube tasks."""
    raise NotImplementedError(
        "so101_cube_rigid_props not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/so101_object_cfg.py"
    )


def so101_cube_spawn_cfg(
    *,
    size: float = SO101_CUBE_SIZE,
    mass: float = SO101_CUBE_MASS,
    contact_offset: float = SO101_CUBE_CONTACT_OFFSET,
    rest_offset: float = SO101_CUBE_REST_OFFSET,
    static_friction: float = SO101_CUBE_STATIC_FRICTION,
    dynamic_friction: float = SO101_CUBE_DYNAMIC_FRICTION,
    restitution: float = SO101_CUBE_RESTITUTION,
    diffuse_color: Sequence[float] = (0.8, 0.5, 0.1),
    metallic: float = 0.0,
) -> "sim_utils.CuboidCfg":
    """Create the canonical 3 cm plastic cube used by SO-101 tasks."""
    raise NotImplementedError(
        "so101_cube_spawn_cfg not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/so101_object_cfg.py"
    )


def so101_cube_object_cfg(
    *,
    prim_path: str,
    init_pos: Sequence[float],
    init_rot: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    diffuse_color: Sequence[float] = (0.8, 0.5, 0.1),
    size: float = SO101_CUBE_SIZE,
) -> "RigidObjectCfg":
    """Create a rigid object cfg with canonical SO-101 cube physics."""
    raise NotImplementedError(
        "so101_cube_object_cfg not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/so101_object_cfg.py"
    )


__all__ = [
    "SO101_CUBE_SIZE",
    "SO101_CUBE_MASS",
    "SO101_CUBE_CONTACT_OFFSET",
    "SO101_CUBE_REST_OFFSET",
    "SO101_CUBE_STATIC_FRICTION",
    "SO101_CUBE_DYNAMIC_FRICTION",
    "SO101_CUBE_RESTITUTION",
    "so101_cube_rigid_props",
    "so101_cube_spawn_cfg",
    "so101_cube_object_cfg",
]
