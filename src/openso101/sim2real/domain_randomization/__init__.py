# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from importlib import import_module

_MODULES = {
    "action": ("ActionDelayBuffer", "FirstOrderActionLag", "attach_action_dr"),
    "observation": ("attach_observation_dr",),
    "visual": ("attach_visual_dr",),
    "wrapper": ("ActionDRWrapper",),
    "physics": (
        "GRAVITY_DISTRIBUTION", "OBJECT_DYNAMIC_FRICTION_RANGE", "OBJECT_MASS_SCALE",
        "OBJECT_NUM_MATERIAL_BUCKETS", "OBJECT_RESTITUTION_RANGE", "OBJECT_STATIC_FRICTION_RANGE",
        "ROBOT_DAMPING_SCALE", "ROBOT_JOINT_ARMATURE_SCALE", "ROBOT_JOINT_FRICTION_SCALE",
        "ROBOT_LINK_MASS_SCALE", "ROBOT_STIFFNESS_SCALE", "attach_all_physics_dr",
        "attach_gravity_dr", "attach_object_physics_dr", "attach_robot_physics_dr",
    ),
}
__all__ = sorted(name for names in _MODULES.values() for name in names)


def __getattr__(name):
    for module, names in _MODULES.items():
        if name in names:
            return getattr(import_module(f".{module}", __name__), name)
    raise AttributeError(name)
