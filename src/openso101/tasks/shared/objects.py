# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Shared SO-101 manipulation object physics."""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

SO101_CUBE_SIZE = 0.03
SO101_CUBE_MASS = 0.02
SO101_CUBE_CONTACT_OFFSET = 0.004
SO101_CUBE_REST_OFFSET = 0.0
SO101_CUBE_STATIC_FRICTION = 1.0
SO101_CUBE_DYNAMIC_FRICTION = 1.0
SO101_CUBE_RESTITUTION = 0.0


def so101_cube_rigid_props(
    *,
    solver_position_iteration_count: int = 16,
    solver_velocity_iteration_count: int = 1,
    max_depenetration_velocity: float = 5.0,
) -> RigidBodyPropertiesCfg:
    """Return rigid-body solver settings shared by SO-101 cube tasks."""

    return RigidBodyPropertiesCfg(
        solver_position_iteration_count=solver_position_iteration_count,
        solver_velocity_iteration_count=solver_velocity_iteration_count,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=max_depenetration_velocity,
        disable_gravity=False,
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
) -> sim_utils.CuboidCfg:
    """Create the canonical 3 cm plastic cube used by SO-101 tasks."""

    return sim_utils.CuboidCfg(
        size=(size, size, size),
        rigid_props=so101_cube_rigid_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=contact_offset,
            rest_offset=rest_offset,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(diffuse_color),
            metallic=metallic,
        ),
    )


def so101_cube_object_cfg(
    *,
    prim_path: str,
    init_pos: Sequence[float],
    init_rot: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    diffuse_color: Sequence[float] = (0.8, 0.5, 0.1),
    size: float = SO101_CUBE_SIZE,
) -> RigidObjectCfg:
    """Create a rigid object cfg with canonical SO-101 cube physics."""

    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(init_pos), rot=list(init_rot)),
        spawn=so101_cube_spawn_cfg(size=size, diffuse_color=diffuse_color),
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
