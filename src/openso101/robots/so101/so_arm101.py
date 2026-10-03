# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""SO-ARM101 robot Isaac Lab articulation config and USD asset resolution."""

# This module resolves and configures the third-party SO-ARM101 USD mesh
# (authored by Muammer Bay (LycheeAI) and Louis Le Lay; see
# LICENSE-BSD-3-CLAUSE for the bundled asset's BSD-3-Clause license). The
# actuator / PD articulation config below is ported "Lior-style" from
# liorbenhorin/lerobot_so101_teleop.

from pathlib import Path
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from openso101.robots.so101.constants import (
    SO101_CANONICAL_INIT_JOINT_POS,
    SO101_GRIPPER_JOINT_NAME,
    SO101_GRIPPER_JOINT_NAMES,
    SO101_GRIPPER_OPEN_POS,
)
from openso101.robots.so101._usd_bounds import tabletop_root_z

REPO_ROOT = Path(__file__).resolve().parents[4]


def so101_usd_path() -> Path:
    """Return the local SO101 USD asset path.

    Resolution order:
    1. ``$OPENSO101_SO101_USD_PATH`` env var if set.
    2. ``<repo_root>/assets/so101/usd/SO-ARM101-USD.usd`` (the default
       fetched location).

    The USD file is **not** committed to the repository (it's a 23 MB
    third-party binary). Run ``scripts/fetch_so101_usd.sh`` after a
    fresh clone, or follow :doc:`/docs/guides/install` ``Step 6``.
    """
    configured = os.environ.get("OPENSO101_SO101_USD_PATH")
    if configured:
        return Path(configured).expanduser()
    return REPO_ROOT / "assets" / "so101" / "usd" / "SO-ARM101-USD.usd"


SO101_USD_TABLETOP_ROOT_Z: float = tabletop_root_z(so101_usd_path())
"""``init_state.pos.z`` that places the SO101 base bottom on a table top at world z=0.

Computed at import time from the USD's base-prim bbox via
:func:`openso101.robots.so101._usd_bounds.tabletop_root_z`.
"""

# 90° yaw about +Z, expressed in Isaac Lab's (w, x, y, z) quaternion order.
# This is the math-clean equivalent of "base 270° + Rotation joint 180°"
# (270° + 180° = 450° = 90°) collapsed onto the base alone, so we don't
# exceed the Rotation joint's ±110° hard limit.
#   w = cos(π/4) ≈ 0.7071068
#   z = sin(π/4) ≈ 0.7071068
SO101_BASE_INIT_ROT: tuple[float, float, float, float] = (
    math.cos(math.pi / 4),
    0.0,
    0.0,
    math.sin(math.pi / 4),
)


# Backwards-compatible alias: legacy code referenced SO101_TELEOP_INIT_JOINT_POS
# directly. Keep the name as a pointer to the unified pose.
SO101_TELEOP_INIT_JOINT_POS = SO101_CANONICAL_INIT_JOINT_POS


# RL-only: a high-friction physics material bound to the gripper / jaw
# collision prims at spawn. The USD's default friction (~0.5) plus the
# cube's friction (1.0) gives an effective grip coefficient too low for the
# RL policy to hold a small cube during exploration. Teleop doesn't use
# this binding — a human operator compensates for slippery jaws by careful
# positioning.
SO101_RL_GRIPPER_STATIC_FRICTION = 1.5
SO101_RL_GRIPPER_DYNAMIC_FRICTION = 1.2


def spawn_so101_usd_with_grip_friction(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the USD as-authored, then bind a high-friction material on
    every authored collider under ``/gripper/collisions`` and
    ``/jaw/collisions``.

    Minimal scope (does NOT touch the authored colliders):
    - No ``CollisionAPI`` applications (previous rewrite did this and
      conflicted with the USD's ``PhysxMeshMergeCollisionAPI``).
    - No merge-API stripping.
    - No collision approximation changes.
    - Only de-instances the collision subtree and binds the friction
      material via ``UsdShade.MaterialBindingAPI`` with
      ``strongerThanDescendants`` so the cube-on-gripper friction uses
      the bound value via ``friction_combine_mode="max"``.

    Used by ``SO_ARM101_CFG`` (RL) only. ``SO_ARM101_TELEOP_CFG`` keeps
    the default ``spawn_from_usd`` (no friction binding) because the
    human-led teleop control loop compensates for slippery jaws and the
    Lior reference config does the same.
    """

    from pxr import Usd, UsdPhysics, UsdShade

    from isaaclab.sim.spawners.from_files import from_files

    prim = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    root_path = str(prim.GetPath())

    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=SO101_RL_GRIPPER_STATIC_FRICTION,
        dynamic_friction=SO101_RL_GRIPPER_DYNAMIC_FRICTION,
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    material_path = f"{root_path}/gripper_contact_material"
    material_cfg.func(material_path, material_cfg)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))

    bound: set[str] = set()
    for group in (f"{root_path}/gripper/collisions", f"{root_path}/jaw/collisions"):
        group_prim = stage.GetPrimAtPath(group)
        if not group_prim or not group_prim.IsValid():
            continue
        # De-instance the subtree so descendant overrides (the material binding
        # below) are authored, not silently dropped on the instance prototype.
        for descendant in Usd.PrimRange(group_prim):
            if descendant.IsInstance():
                descendant.SetInstanceable(False)
        for descendant in Usd.PrimRange(group_prim):
            path = str(descendant.GetPath())
            if path in bound or not descendant.HasAPI(UsdPhysics.CollisionAPI):
                continue
            bound.add(path)
            binding = UsdShade.MaterialBindingAPI.Apply(descendant)
            binding.Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )

    return prim


##
# Configuration
##

# Aggressive Lior-style RL config. Mirrors SO_ARM101_TELEOP_CFG below in PD
# gains, solver settings, and effort caps. Three intentional divergences:
#   1. activate_contact_sensors=True — required for grasped_reward.
#   2. velocity_limit_sim=SO101_RL_VELOCITY_LIMIT — caps the compliant
#      high-effort actuator from slamming toward policy-issued targets.
#   3. Custom spawn func binds a high-friction material on gripper colliders
#      so the policy can grip a small cube during exploration; teleop's
#      slipperier USD-default friction is fine for human-led grasping.
SO101_RL_VELOCITY_LIMIT = 2.0
SO_ARM101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_so101_usd_with_grip_friction,
        usd_path=str(so101_usd_path()),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, SO101_USD_TABLETOP_ROOT_Z),
        rot=SO101_BASE_INIT_ROT,
        joint_pos=SO101_CANONICAL_INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit_sim=30,
            velocity_limit_sim=SO101_RL_VELOCITY_LIMIT,
            stiffness=55,
            damping=0.7,
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit_sim=30,
            velocity_limit_sim=SO101_RL_VELOCITY_LIMIT,
            stiffness=30,
            damping=0.8,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit_sim=30,
            velocity_limit_sim=SO101_RL_VELOCITY_LIMIT,
            stiffness=25,
            damping=0.7,
        ),
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit_sim=30,
            velocity_limit_sim=SO101_RL_VELOCITY_LIMIT,
            stiffness=12,
            damping=0.5,
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit_sim=30,
            velocity_limit_sim=SO101_RL_VELOCITY_LIMIT,
            stiffness=7,
            damping=0.5,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=list(SO101_GRIPPER_JOINT_NAMES),
            effort_limit_sim=30,
            velocity_limit_sim=SO101_RL_VELOCITY_LIMIT,
            stiffness=15,
            damping=0.5,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)


##
# Teleop variant
##

# Independent robot config for hand-teleoperation scenes. Verbatim port of
# liorbenhorin/lerobot_so101_teleop's SO101_CFG approach (assets/so101.py)
# with our scene-specific bits (init_pos on the table top, fix_root_link)
# preserved. See the SO_ARM101_CFG block above for the three intentional
# divergences (contact sensors, velocity cap, friction-binding spawn);
# teleop trusts the USD's authored colliders + friction verbatim because
# human-led control compensates for slippery jaws.
SO_ARM101_TELEOP_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(so101_usd_path()),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, SO101_USD_TABLETOP_ROOT_Z),
        rot=SO101_BASE_INIT_ROT,
        joint_pos=SO101_TELEOP_INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit_sim=30,
            stiffness=55,
            damping=0.7,
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit_sim=30,
            stiffness=30,
            damping=0.8,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit_sim=30,
            stiffness=25,
            damping=0.7,
        ),
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit_sim=30,
            stiffness=12,
            damping=0.5,
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit_sim=30,
            stiffness=7,
            damping=0.5,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=list(SO101_GRIPPER_JOINT_NAMES),
            effort_limit_sim=30,
            stiffness=4,
            damping=0.3,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
"""Independent teleop SO-101 articulation config. Used by teleop scene cfgs.
Do not import this into RL tasks — the RL canonical is SO_ARM101_CFG."""
