from pathlib import Path
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

SO101_GRIPPER_CONTACT_STATIC_FRICTION = 1.2
SO101_GRIPPER_CONTACT_DYNAMIC_FRICTION = 1.0
SO101_GRIPPER_CONTACT_OFFSET = 0.002
SO101_GRIPPER_REST_OFFSET = 0.0


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
    "Rotation": 1.5708,    # +π/2 — base yawed LEFT 90° so the arm faces the
                           # cube spawn (at +X in the world; with the SO-101
                           # base zero pointing along +Y, +π/2 rotates the
                           # arm's front to +X).
    "Pitch": 0.0,
    "Elbow": 0.0,
    "Wrist_Pitch": 1.5708,  # π/2 — gripper pointing straight down at table
    "Wrist_Roll": 0.0,
    SO101_GRIPPER_JOINT_NAME: SO101_GRIPPER_OPEN_POS,
}
"""Canonical SO101 reset posture used by both RL training and teleop.

The arm starts upright with the base yawed +π/2 so the gripper points
straight at the cube spawn (0.3, 0, 0.015), and the gripper is pointed
straight down so it can descend onto the cube. Used by both
``SO_ARM101_CFG`` (RL) and ``SO_ARM101_TELEOP_CFG`` (teleop) — the same
"face the cube" pose is the right default for both pillars (a trained
policy moves away from any init pose within a few timesteps; teleop
operators want minimal lurch from their leader's home calibration; both
favor a forward-facing rest).
"""


# Backwards-compatible alias: legacy code referenced SO101_TELEOP_INIT_JOINT_POS
# directly. Keep the name as a pointer to the unified pose.
SO101_TELEOP_INIT_JOINT_POS = SO101_CANONICAL_INIT_JOINT_POS


def spawn_so101_usd_with_safe_collisions(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn SO101 USD while avoiding GPU-SDF gripper contacts.

    The source asset authors the gripper and jaw collision groups as SDF meshes.
    That path is fragile for GPU PhysX on this small grasp task, so those two
    contact groups are de-instanced, cooked as convex hulls, and assigned
    explicit contact offsets plus high-friction contact material at spawn time.
    """

    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade

    from isaaclab.sim.spawners.from_files import from_files

    prim = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    root_path = str(prim.GetPath())
    contact_groups = (f"{root_path}/gripper/collisions", f"{root_path}/jaw/collisions")
    contact_material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=SO101_GRIPPER_CONTACT_STATIC_FRICTION,
        dynamic_friction=SO101_GRIPPER_CONTACT_DYNAMIC_FRICTION,
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    contact_material_path = f"{root_path}/gripper_contact_material"
    contact_material_cfg.func(contact_material_path, contact_material_cfg)
    contact_material = UsdShade.Material(stage.GetPrimAtPath(contact_material_path))
    configured_paths: set[str] = set()

    def make_collision_tree_editable(group_prim):
        if group_prim.IsInstance():
            group_prim.SetInstanceable(False)
        for descendant in Usd.PrimRange(group_prim):
            if descendant.IsInstance():
                descendant.SetInstanceable(False)

    def configure_contact_prim(contact_prim):
        contact_path = str(contact_prim.GetPath())
        if contact_path in configured_paths or not contact_prim.HasAPI(UsdPhysics.CollisionAPI):
            return
        configured_paths.add(contact_path)

        approx_attr = contact_prim.GetAttribute("physics:approximation")
        if not approx_attr:
            approx_attr = contact_prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token)
        approx_attr.Set("convexHull")

        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(contact_prim)
        physx_collision.CreateContactOffsetAttr().Set(SO101_GRIPPER_CONTACT_OFFSET)
        physx_collision.CreateRestOffsetAttr().Set(SO101_GRIPPER_REST_OFFSET)
        binding_api = UsdShade.MaterialBindingAPI.Apply(contact_prim)
        binding_api.Bind(
            contact_material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )

    for group in contact_groups:
        group_prim = stage.GetPrimAtPath(group)
        if group_prim:
            make_collision_tree_editable(group_prim)

    for child in Usd.PrimRange(prim):
        child_path = str(child.GetPath())
        if not any(child_path == group or child_path.startswith(f"{group}/") for group in contact_groups):
            continue
        configure_contact_prim(child)

    # Direct paths are the authored collision API prims in the upstream asset.
    # Keep this explicit fallback so a future USD hierarchy change is obvious.
    for group in contact_groups:
        group_prim = stage.GetPrimAtPath(group)
        if group_prim:
            configure_contact_prim(group_prim)

    return prim

##
# Configuration
##

SO_ARM101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_so101_usd_with_safe_collisions,
        usd_path=str(so101_usd_path()),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # URDF-era settings restored. enabled_self_collisions=True is more
            # physically accurate; solver iter 8/0 matches the URDF cfg and is
            # cheaper than 32/1; fix_root_link=True was added in a prior commit
            # so the base sits on the table top (URDF had fix_base=True on the
            # spawner). soft_joint_pos_limit_factor lives on ArticulationCfg
            # itself, not here.
            enabled_self_collisions=True,
            fix_root_link=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, SO101_USD_TABLETOP_ROOT_Z),
        joint_pos=SO101_CANONICAL_INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        # URDF-era gains restored to the USD canonical (2026-05-13 swap regressed
        # these to overly-weak hand-tuned values that prevented the gripper from
        # closing and let the arm whip around unbounded). Joint names map:
        # shoulder_pan -> Rotation, shoulder_lift -> Pitch, elbow_flex -> Elbow,
        # wrist_flex -> Wrist_Pitch, wrist_roll -> Wrist_Roll, gripper -> Jaw.
        # Source: commit 5d01353^:src/safe_sim2real/robots/trs_so101/so_arm101.py.
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit_sim=1.9,
            velocity_limit_sim=1.5,
            stiffness=200.0,
            damping=80.0,
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit_sim=1.9,
            velocity_limit_sim=1.5,
            stiffness=170.0,
            damping=65.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit_sim=1.9,
            velocity_limit_sim=1.5,
            stiffness=120.0,
            damping=45.0,
        ),
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit_sim=1.9,
            velocity_limit_sim=1.5,
            stiffness=80.0,
            damping=30.0,
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit_sim=1.9,
            velocity_limit_sim=1.5,
            stiffness=50.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=list(SO101_GRIPPER_JOINT_NAMES),
            effort_limit_sim=2.5,
            velocity_limit_sim=1.5,
            stiffness=60.0,
            damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)


##
# Teleop variant
##

# Independent robot config for hand-teleoperation scenes. Compared to the RL
# canonical, teleop needs:
# - Higher actuator effort + velocity caps so the follower tracks a human leader
#   arm responsively (RL caps are deliberately tame for safe exploration).
# - enabled_self_collisions=False because human teleop can drive the leader
#   into awkward configurations; without this the follower binds up.
# - Higher solver iterations + lower depenetration velocity for stable
#   hand-tracking contacts during fast user motion.
# Stiffness and damping match the RL canonical (URDF-era proven values).
SO_ARM101_TELEOP_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_so101_usd_with_safe_collisions,
        usd_path=str(so101_usd_path()),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, SO101_USD_TABLETOP_ROOT_Z),
        joint_pos=SO101_TELEOP_INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit_sim=8.0,
            velocity_limit_sim=3.5,
            stiffness=200.0,
            damping=80.0,
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit_sim=8.0,
            velocity_limit_sim=3.5,
            stiffness=170.0,
            damping=65.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit_sim=8.0,
            velocity_limit_sim=3.5,
            stiffness=120.0,
            damping=45.0,
        ),
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit_sim=4.0,
            velocity_limit_sim=2.0,
            stiffness=80.0,
            damping=30.0,
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit_sim=4.0,
            velocity_limit_sim=2.0,
            stiffness=50.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=list(SO101_GRIPPER_JOINT_NAMES),
            effort_limit_sim=4.0,
            velocity_limit_sim=2.0,
            stiffness=60.0,
            damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
"""Independent teleop SO-101 articulation config. Used by teleop scene cfgs.
Do not import this into RL tasks — the RL canonical is SO_ARM101_CFG."""
