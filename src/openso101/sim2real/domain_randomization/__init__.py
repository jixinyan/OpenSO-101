# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Domain randomization layer.

Three production modules: ``physics.py`` (link masses, joint
friction/armature, actuator gains, object mass + contact material,
gravity), ``visual.py`` (dome-light intensity/color, object diffuse
color), and ``observation.py`` (joint-position + joint-velocity noise).

``action.py`` is an intentional stub — wrapping action terms config-
time requires a custom ``ActionTerm`` subclass that Isaac Lab 2.3.0
doesn't expose. It raises ``NotImplementedError`` if called, but is
not part of the public ``__all__`` so ``import *`` doesn't surface
it.
"""

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
    # Physics DR — constants
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
    # Physics DR — functions
    "attach_robot_physics_dr",
    "attach_object_physics_dr",
    "attach_gravity_dr",
    "attach_all_physics_dr",
    # Visual DR
    "attach_visual_dr",
    # Observation DR
    "attach_observation_dr",
]
