# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Shared domain-randomization (DR) config for SO-ARM101 tasks.

The goal is to make policies trained in sim robust to the physics and sensor
uncertainties of the real SO-ARM101 rig.  DR is layered in phases so each
layer can be validated independently:

* **Phase 1 — physics DR (this file).** Randomize link masses, actuator PD
  gains, joint friction, object mass and contact materials, and gravity via
  ``EventTerm``s on each task's ``EventCfg``.
* **Phase 2 — action/observation DR (planned).** Add conservative joint-state
  noise, action noise, and bounded action latency before any visual DR so both
  state policies and future vision policies share the same control robustness.
* **Phase 3 — visual DR (planned).** Add camera pose jitter, lighting, and
  appearance variation only to ``-Vision`` task variants so state PPO training
  stays fast.

Keep ranges centralized here and task wiring explicit in each task config.
Widen ranges for more robust but slower-converging policies; tighten them
when debugging convergence.

Usage in a task ``__post_init__``::

    from openso101.sim2real.domain_randomization.physics import (
        attach_robot_physics_dr, attach_object_physics_dr, attach_gravity_dr,
    )

    attach_robot_physics_dr(self.events, robot_asset_name="robot")
    attach_object_physics_dr(self.events, object_asset_name="cube_top")
    attach_object_physics_dr(self.events, object_asset_name="cube_bottom")
    attach_gravity_dr(self.events)
"""

from __future__ import annotations

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

##
# Tunable ranges — edit here to re-calibrate DR for all tasks at once.
##

# --- Robot dynamics (applied at startup: fixed per training run per env) ---
ROBOT_LINK_MASS_SCALE = (0.8, 1.2)           # ±20% on every link mass
ROBOT_JOINT_FRICTION_SCALE = (0.5, 2.0)      # wide: joint friction is hard to measure
ROBOT_JOINT_ARMATURE_SCALE = (0.8, 1.2)      # rotor inertia scale

# --- Actuator gains (applied at reset: re-sampled each episode) ---
ROBOT_STIFFNESS_SCALE = (0.75, 1.25)         # ±25% on PD stiffness
ROBOT_DAMPING_SCALE = (0.75, 1.25)           # ±25% on PD damping

# --- Object physics (applied at startup for materials, reset for mass) ---
OBJECT_MASS_SCALE = (0.7, 1.3)               # ±30% on object mass
OBJECT_STATIC_FRICTION_RANGE = (0.5, 1.5)    # absolute friction coefficients
OBJECT_DYNAMIC_FRICTION_RANGE = (0.3, 1.2)
OBJECT_RESTITUTION_RANGE = (0.0, 0.3)
OBJECT_NUM_MATERIAL_BUCKETS = 128            # PhysX caps unique materials ~64000

# --- Scene-level (applied at startup) ---
# Additive noise on gravity vector (m/s²). The real rig is on Earth so we
# keep this small; mostly captures inclined-table / IMU-calibration drift.
GRAVITY_DISTRIBUTION = ([0.0, 0.0, -0.2], [0.0, 0.0, 0.2])


##
# Helpers that attach DR terms to an existing EventCfg instance.
##


def attach_robot_physics_dr(events, robot_asset_name: str = "robot") -> None:
    """Attach link mass, joint friction/armature, and actuator gain randomization."""

    events.dr_robot_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(robot_asset_name, body_names=".*"),
            "mass_distribution_params": ROBOT_LINK_MASS_SCALE,
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # Isaac Lab 2.3 has a tensor-shape bug in ``randomize_joint_parameters``
    # when ``joint_ids=slice(None)``; force an explicit joint list with
    # ``joint_names=".*"`` so the advanced-indexing path is used.
    events.dr_robot_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(robot_asset_name, joint_names=".*"),
            "friction_distribution_params": ROBOT_JOINT_FRICTION_SCALE,
            "armature_distribution_params": ROBOT_JOINT_ARMATURE_SCALE,
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    events.dr_robot_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(robot_asset_name, joint_names=".*"),
            "stiffness_distribution_params": ROBOT_STIFFNESS_SCALE,
            "damping_distribution_params": ROBOT_DAMPING_SCALE,
            "operation": "scale",
            "distribution": "uniform",
        },
    )


def attach_object_physics_dr(events, object_asset_name: str) -> None:
    """Attach mass and contact-material randomization to a rigid object."""

    setattr(
        events,
        f"dr_{object_asset_name}_mass",
        EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(object_asset_name),
                "mass_distribution_params": OBJECT_MASS_SCALE,
                "operation": "scale",
                "distribution": "uniform",
            },
        ),
    )
    setattr(
        events,
        f"dr_{object_asset_name}_material",
        EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(object_asset_name),
                "static_friction_range": OBJECT_STATIC_FRICTION_RANGE,
                "dynamic_friction_range": OBJECT_DYNAMIC_FRICTION_RANGE,
                "restitution_range": OBJECT_RESTITUTION_RANGE,
                "num_buckets": OBJECT_NUM_MATERIAL_BUCKETS,
            },
        ),
    )


def attach_gravity_dr(events) -> None:
    """Attach small additive gravity noise."""

    events.dr_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="startup",
        params={
            "gravity_distribution_params": GRAVITY_DISTRIBUTION,
            "operation": "add",
            "distribution": "uniform",
        },
    )


def attach_all_physics_dr(
    events,
    robot_asset_name: str = "robot",
    object_asset_names: tuple[str, ...] = (),
    include_gravity: bool = True,
) -> None:
    """Convenience: attach every Phase-1 physics DR term in one call."""

    attach_robot_physics_dr(events, robot_asset_name=robot_asset_name)
    for name in object_asset_names:
        attach_object_physics_dr(events, object_asset_name=name)
    if include_gravity:
        attach_gravity_dr(events)


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
