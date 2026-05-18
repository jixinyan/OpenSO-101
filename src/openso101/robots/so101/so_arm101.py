from pathlib import Path
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from openso101.robots.so101.constants import (
    SO101_DEFAULT_JOINT_POS,
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


##
# Configuration
##

# Aggressive Lior-style RL config. Mirrors SO_ARM101_TELEOP_CFG below in PD
# gains, solver settings, and effort caps. Two intentional divergences:
#
#   1. activate_contact_sensors=True (teleop is False). Required for
#      pick_place's grasped_reward, which reads /Robot/gripper and
#      /Robot/jaw contact signals.
#   2. velocity_limit_sim=SO101_RL_VELOCITY_LIMIT on every actuator (teleop
#      leaves it unset, matching Lior verbatim). Without this cap, the
#      compliant high-effort actuator slams toward any policy-issued
#      position target — observed symptom was the arm whipping at
#      ~25 rad/s during RL. Teleop doesn't need the cap because the
#      leader-arm-driven targets are naturally bounded by human motion.
SO101_RL_VELOCITY_LIMIT = 2.0
SO_ARM101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
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
        # Stiffness bumped from Lior's k=4 to k=15 for RL only. With k=4, the
        # jaws close too slowly to pin a moving cube during policy exploration:
        # the policy can never assign credit to "close gripper" because by the
        # time the jaws close, the cube has already moved. Teleop stays at k=4
        # because human-driven targets give enough lead time. k=15 is still
        # well below the URDF-era 60 that "hammered" the cube.
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
# preserved. Key differences from the RL canonical:
# - No custom collision spawn function. Trust the USD's authored colliders.
#   Earlier attempts to add standalone CollisionAPI on merge children
#   silently disabled gripper collision; Lior trusts the USD and it grips.
# - activate_contact_sensors=False. Teleop has no contact-gated rewards;
#   contact sensors only add cost.
# - Compliant low-stiffness PD gains (e.g. gripper k=4, d=0.3 vs RL's
#   k=60, d=20). Matches real SO-101 servo bandwidth; lets the jaws pinch
#   rather than hammer. effort_limit_sim=30 across all joints provides
#   headroom for the compliant gains to actually reach the target.
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
