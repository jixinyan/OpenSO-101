# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Phase 1 physics domain randomization for SO-101 manipulation tasks.

SKELETON: real range constants + function bodies will be ported from
`/data/safe_sim2real/src/safe_sim2real/tasks/domain_randomization.py`
once the source is finalized.

The module covers robot link masses, joint friction + armature, actuator
PD gains, per-object mass + contact material, and small gravity noise.
Ranges live as module-level constants so they are easy to tune.
"""

from __future__ import annotations

# --- Robot dynamics (applied at startup: fixed per training run per env) ---

# TODO: port concrete tuples from legacy source.
ROBOT_LINK_MASS_SCALE: tuple[float, float] = (0.0, 0.0)
ROBOT_JOINT_FRICTION_SCALE: tuple[float, float] = (0.0, 0.0)
ROBOT_JOINT_ARMATURE_SCALE: tuple[float, float] = (0.0, 0.0)

# --- Actuator gains (applied at reset: re-sampled each episode) ---
ROBOT_STIFFNESS_SCALE: tuple[float, float] = (0.0, 0.0)
ROBOT_DAMPING_SCALE: tuple[float, float] = (0.0, 0.0)

# --- Object physics (applied at startup for materials, reset for mass) ---
OBJECT_MASS_SCALE: tuple[float, float] = (0.0, 0.0)
OBJECT_STATIC_FRICTION_RANGE: tuple[float, float] = (0.0, 0.0)
OBJECT_DYNAMIC_FRICTION_RANGE: tuple[float, float] = (0.0, 0.0)
OBJECT_RESTITUTION_RANGE: tuple[float, float] = (0.0, 0.0)
OBJECT_NUM_MATERIAL_BUCKETS: int = 0  # TODO: port concrete value from legacy source.

# --- Scene-level (applied at startup) ---
# Additive noise on gravity vector (m/s²).
GRAVITY_DISTRIBUTION: tuple[list[float], list[float]] = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def attach_robot_physics_dr(events, robot_asset_name: str = "robot") -> None:
    """Attach link mass, joint friction/armature, and actuator gain randomization.

    SKELETON: real implementation will be ported from
    /data/safe_sim2real/src/safe_sim2real/tasks/domain_randomization.py.
    """
    raise NotImplementedError(
        "attach_robot_physics_dr not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/domain_randomization.py"
    )


def attach_object_physics_dr(events, object_asset_name: str) -> None:
    """Attach mass and contact-material randomization to a rigid object.

    SKELETON: real implementation will be ported from
    /data/safe_sim2real/src/safe_sim2real/tasks/domain_randomization.py.
    """
    raise NotImplementedError(
        "attach_object_physics_dr not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/domain_randomization.py"
    )


def attach_gravity_dr(events) -> None:
    """Attach small additive gravity noise.

    SKELETON: real implementation will be ported from
    /data/safe_sim2real/src/safe_sim2real/tasks/domain_randomization.py.
    """
    raise NotImplementedError(
        "attach_gravity_dr not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/domain_randomization.py"
    )


def attach_all_physics_dr(
    events,
    robot_asset_name: str = "robot",
    object_asset_names: tuple[str, ...] = (),
    include_gravity: bool = True,
) -> None:
    """Convenience: attach every Phase-1 physics DR term in one call.

    SKELETON: real implementation will be ported from
    /data/safe_sim2real/src/safe_sim2real/tasks/domain_randomization.py.
    """
    raise NotImplementedError(
        "attach_all_physics_dr not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/domain_randomization.py"
    )


__all__ = [
    # Constants
    "ROBOT_LINK_MASS_SCALE",
    "ROBOT_JOINT_FRICTION_SCALE",
    "ROBOT_JOINT_ARMATURE_SCALE",
    "ROBOT_STIFFNESS_SCALE",
    "ROBOT_DAMPING_SCALE",
    "OBJECT_MASS_SCALE",
    "OBJECT_STATIC_FRICTION_RANGE",
    "OBJECT_DYNAMIC_FRICTION_RANGE",
    "OBJECT_RESTITUTION_RANGE",
    "OBJECT_NUM_MATERIAL_BUCKETS",
    "GRAVITY_DISTRIBUTION",
    # Functions
    "attach_robot_physics_dr",
    "attach_object_physics_dr",
    "attach_gravity_dr",
    "attach_all_physics_dr",
]
