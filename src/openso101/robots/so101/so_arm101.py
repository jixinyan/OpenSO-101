from pathlib import Path
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from openso101.robots.so101.constants import (
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
:func:`openso101.robots.so101._usd_bounds.tabletop_root_z`. When
pxr is not available (e.g. unit tests without the Omniverse app), falls
back to a baked constant; see ``_usd_bounds._BAKED_BASE_PRIM_LOCAL_Z_MIN``.
"""

SO101_CANONICAL_INIT_JOINT_POS: dict[str, float] = {
    # All horizontal rotation is expressed on the base (SO101_BASE_INIT_ROT)
    # so we don't fight the Rotation joint's ±110° hard limit on reset.
    # Joint stays at zero — base yaw alone determines where the arm points.
    "Rotation": 0.0,
    "Pitch": 0.0,
    "Elbow": 0.0,
    "Wrist_Pitch": 1.5708,  # π/2 — gripper pointing straight down at table.
    "Wrist_Roll": 0.0,
    SO101_GRIPPER_JOINT_NAME: SO101_GRIPPER_OPEN_POS,
}
"""Canonical SO101 reset posture used by both RL training and teleop.

Identity base orientation; arm joints at zero except Wrist_Pitch (π/2)
which points the gripper straight down at the table. The Rotation joint
is intentionally zero — the base sits in its USD-canonical orientation
with the arm extending along the base's natural front. RL policies move
away from this pose within a few timesteps; teleop operators see the
arm at its mechanically neutral pose with the gripper ready to descend.
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
        # ROTATION (Gear: 1/191, Torque: 34.4 N-m)
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit_sim=30,
            velocity_limit_sim=SO101_RL_VELOCITY_LIMIT,
            stiffness=55,
            damping=0.7,
        ),
        # PITCH (Gear: 1/345, Torque: 62.1 N-m - HIGHEST)
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit_sim=30,
            velocity_limit_sim=SO101_RL_VELOCITY_LIMIT,
            stiffness=30,
            damping=0.8,
        ),
        # ELBOW (Gear: 1/191, Torque: 34.4 N-m)
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit_sim=30,
            velocity_limit_sim=SO101_RL_VELOCITY_LIMIT,
            stiffness=25,
            damping=0.7,
        ),
        # WRIST PITCH (Gear: 1/147, Torque: 26.5 N-m)
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit_sim=30,
            velocity_limit_sim=SO101_RL_VELOCITY_LIMIT,
            stiffness=12,
            damping=0.5,
        ),
        # WRIST ROLL (Gear: 1/147, Torque: 26.5 N-m)
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit_sim=30,
            velocity_limit_sim=SO101_RL_VELOCITY_LIMIT,
            stiffness=7,
            damping=0.5,
        ),
        # GRIPPER (Gear: 1/147, Torque: 26.5 N-m)
        # Stiffness bumped from Lior's k=4 (teleop) to k=15 for RL: k=4 closes
        # too slowly to pin a moving cube during exploration, so the policy
        # never gets credit for "close gripper". Still well below the URDF-era
        # k=60 that hammered the cube.
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
        # ROTATION (Gear: 1/191, Torque: 34.4 N-m)
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit_sim=30,
            stiffness=55,
            damping=0.7,
        ),
        # PITCH (Gear: 1/345, Torque: 62.1 N-m - HIGHEST)
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit_sim=30,
            stiffness=30,
            damping=0.8,
        ),
        # ELBOW (Gear: 1/191, Torque: 34.4 N-m)
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit_sim=30,
            stiffness=25,
            damping=0.7,
        ),
        # WRIST PITCH (Gear: 1/147, Torque: 26.5 N-m)
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit_sim=30,
            stiffness=12,
            damping=0.5,
        ),
        # WRIST ROLL (Gear: 1/147, Torque: 26.5 N-m)
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit_sim=30,
            stiffness=7,
            damping=0.5,
        ),
        # GRIPPER (Gear: 1/147, Torque: 26.5 N-m)
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
