# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Domain randomization layer.

Phase 1 (physics DR) lives in ``physics.py`` and is the only fully ported
sub-module today: link masses, joint friction/armature, actuator gains,
object mass + contact material, and small additive gravity noise.

Phase 2 (action/observation DR) and Phase 3 (visual DR) are tracked by the
``action.py`` / ``observation.py`` / ``visual.py`` skeletons; those modules
remain stubbed pending sub-project B.
"""

from .action import attach_action_dr
from .observation import attach_observation_dr
from .physics import (
    GRAVITY_DISTRIBUTION,
    OBJECT_DYNAMIC_FRICTION_RANGE,
    OBJECT_MASS_SCALE,
    OBJECT_NUM_MATERIAL_BUCKETS,
    OBJECT_RESTITUTION_RANGE,
    OBJECT_STATIC_FRICTION_RANGE,
    ROBOT_DAMPING_SCALE,
    ROBOT_JOINT_ARMATURE_SCALE,
    ROBOT_JOINT_FRICTION_SCALE,
    ROBOT_LINK_MASS_SCALE,
    ROBOT_STIFFNESS_SCALE,
    attach_all_physics_dr,
    attach_gravity_dr,
    attach_object_physics_dr,
    attach_robot_physics_dr,
)
from .visual import attach_visual_dr

__all__ = [
    # Phase 1 physics DR — constants
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
    # Phase 1 physics DR — functions
    "attach_robot_physics_dr",
    "attach_object_physics_dr",
    "attach_gravity_dr",
    "attach_all_physics_dr",
    # Phase 2/3 stubs (raise NotImplementedError)
    "attach_action_dr",
    "attach_observation_dr",
    "attach_visual_dr",
]
