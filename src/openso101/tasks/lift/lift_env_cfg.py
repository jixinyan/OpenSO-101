# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Lift task -- pick up a cube from the table.

Single-class env cfg with variant hooks. Collapses legacy `LiftEnvCfg`,
`SoArm101LiftCubeEnvCfg`, `_PLAY`, `WithCameras`, and teleop variants from
`safe_sim2real.tasks.composite.lift.{lift_env_cfg, joint_pos_env_cfg}` into
a single `OpenSO101EnvCfg` subclass whose `configure_*` hooks mutate the
cfg in-place per `gym.make()` kwarg.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import openso101.tasks.lift.mdp as mdp
from openso101.envs import OpenSO101EnvCfg, TeleopActionsCfg, UnsupportedVariantError
from openso101.robots import (
    SO101_ARM_JOINT_NAMES,
    SO101_GRIPPER_CLOSED_POS,
    SO101_GRIPPER_JOINT_NAME,
    SO101_GRIPPER_JOINT_NAMES,
    SO101_GRIPPER_OPEN_POS,
)
from openso101.robots.so101.so_arm101 import SO_ARM101_CFG, SO_ARM101_TELEOP_CFG
from openso101.tasks.shared.objects import so101_cube_object_cfg
from openso101.tasks.shared.rl_defaults import (
    SO101_ACTION_RATE_WEIGHT,
    SO101_CONTROLLED_OBJECT_MIN_HEIGHT,
    SO101_GOAL_TRACKING_STD,
    SO101_JOINT_POS_DELTA_WEIGHT,
    SO101_JOINT_VEL_WEIGHT,
    SO101_REACH_REWARD_COARSE_STD,
    SO101_REACH_REWARD_COARSE_WEIGHT,
    SO101_REACH_REWARD_STD,
    SO101_SMOOTHNESS_CURRICULUM_STEPS,
    SO101_SMOOTHNESS_CURRICULUM_WEIGHT,
)


##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object."""

    # robots: will be populated by env cfg __post_init__
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by env cfg __post_init__
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by env cfg __post_init__
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Contact sensors on the two opposing jaw bodies, filtered to the cube prim.
    # Feed the ``grasped`` reward (contact-confirmed pinch). Optional because
    # teleop strips them — the teleop robot config sets activate_contact_sensors
    # to False and the bodies have no contact reporter API.
    gripper_jaw_contact: ContactSensorCfg | None = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    moving_jaw_contact: ContactSensorCfg | None = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.CubeAbovePoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by env cfg __post_init__
        object_name="object",
        # Per-env lift height above the cube; goal sphere ends up 15-20 cm
        # above the cube's top face (cube center is at world z = 0.015).
        lift_height_range=(0.15, 0.20),
        # resampling_time_range == episode_length_s -> resample fires once per
        # episode, right after the cube-reset event; goal is locked to the
        # initial cube xy.
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        # Visible green sphere at the goal (radius matches lift_success
        # goal_radius). Replaces the default RGB axis triad, which is hard
        # to spot in saved training videos.
        goal_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "goal": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.2)),
                ),
            },
        ),
        # pos_* ranges are ignored by CubeAbovePoseCommand; only roll/pitch/yaw
        # are read (defaults -> identity quat).
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # No ObsTerm in this group sets a noise=Unoise/Gnoise, so observation
            # corruption is currently a no-op. Set False to avoid the misleading
            # "I think I'm noisy" state; re-enable only when per-term noise is
            # intentionally configured.
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    """Minimal three-signal reward chain: reach -> lift -> goal-track-in-air.

    No close_gripper shaping, no "controlled" gates. The lift reward (height-
    only) implicitly rewards grasping because lifting the cube requires
    closing the gripper. Goal-tracking is height-gated so it only fires once
    the cube is airborne. Smoothness penalties start at 0 and curriculum-ramp
    in once lift fires.
    """

    # Two-scale reach: coarse covers ~35cm starting distance, fine sharpens
    # at grasp range.
    reaching_object_coarse = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": SO101_REACH_REWARD_COARSE_STD},
        weight=SO101_REACH_REWARD_COARSE_WEIGHT,
    )
    reaching_object_fine = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": SO101_REACH_REWARD_STD},
        weight=1.0,
    )

    # Contact-confirmed grasp bonus (both jaws register force > threshold).
    # Dense gradient for the close-then-lift sequence; without it the policy
    # stalls on "just reach" before entropy collapses.
    grasped = RewTerm(
        func=mdp.grasped_reward,
        params={"force_threshold": 0.5},
        weight=3.0,
    )

    # Height-only lift reward. No AND-conjunction with gripper-closed or
    # EE-near. Once the cube goes up, this fires.
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": SO101_CONTROLLED_OBJECT_MIN_HEIGHT},
        weight=3.0,
    )

    # Height-only goal tracking, same minimal-height gate as the lift term.
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": SO101_GOAL_TRACKING_STD,
            "minimal_height": SO101_CONTROLLED_OBJECT_MIN_HEIGHT,
            "command_name": "object_pose",
        },
        weight=5.0,
    )

    # Sparse per-step bonus when the cube enters the goal region while
    # still in the air. goal_radius matches the lift_success termination
    # (0.05 m) so the reward fires inside the same success ball.
    success_bonus_in_air = RewTerm(
        func=mdp.object_reached_goal_in_air,
        params={
            "minimal_height": SO101_CONTROLLED_OBJECT_MIN_HEIGHT,
            "goal_radius": 0.05,
            "command_name": "object_pose",
        },
        weight=3.0,
    )

    # Smoothness (zero-weighted initially; curriculum ramps them in
    # once lift fires).
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=SO101_ACTION_RATE_WEIGHT)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=SO101_JOINT_VEL_WEIGHT,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_pos_delta = RewTerm(
        func=mdp.joint_pos_delta_l2,
        weight=SO101_JOINT_POS_DELTA_WEIGHT,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )
    lift_success = DoneTerm(
        func=mdp.lift_success_height_only,
        params={
            "minimal_height": SO101_CONTROLLED_OBJECT_MIN_HEIGHT,
            "goal_radius": 0.05,
            "command_name": "object_pose",
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.

    NOTE: joint_vel is NOT in the curriculum — it's active from step 0
    (see SO101_JOINT_VEL_WEIGHT). Only exploration-suppressing penalties
    we want delayed until lift fires (action_rate) belong here.
    """

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "action_rate",
            "weight": SO101_SMOOTHNESS_CURRICULUM_WEIGHT,
            "num_steps": SO101_SMOOTHNESS_CURRICULUM_STEPS,
        },
    )


##
# Environment configuration
##


def _configure_so101_lift_scene(cfg: "LiftEnvCfg", robot_cfg=None) -> None:
    """Populate the SO-101 robot, cube, and EE frame on a Lift scene."""
    if robot_cfg is None:
        robot_cfg = SO_ARM101_CFG
    cfg.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cfg.scene.object = so101_cube_object_cfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_pos=[0.2, 0.0, 0.015],
    )

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    cfg.scene.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=True,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper",
                name="end_effector",
                offset=OffsetCfg(pos=[0.01, 0.0, -0.09]),
            ),
        ],
    )


@configclass
class LiftEnvCfg(OpenSO101EnvCfg):
    """Single-class Lift cfg with variant hooks."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # Base RL setup (from LiftEnvCfg + SoArm101LiftCubeEnvCfg).
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        # 4096-env training with the Lior-style 32-iteration solver generates
        # ~340 MB of contact pairs per step; the default ~64 MB buffer
        # overflows and PhysX silently drops contacts. Sized at 512 MB for
        # headroom; drop to 256 MB if running with num_envs <= 1024.
        self.sim.physx.gpu_collision_stack_size = 512 * 1024 * 1024

        # SO-101 scene wiring.
        _configure_so101_lift_scene(self)

        # Actions: arm joint-pos (delta scale) + gripper binary toggle.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(SO101_ARM_JOINT_NAMES),
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(SO101_GRIPPER_JOINT_NAMES),
            open_command_expr={SO101_GRIPPER_JOINT_NAME: SO101_GRIPPER_OPEN_POS},
            close_command_expr={SO101_GRIPPER_JOINT_NAME: SO101_GRIPPER_CLOSED_POS},
        )

        # Set the body name for the end effector command target.
        self.commands.object_pose.body_name = ["gripper"]

        # Physics DR (robot link mass + joint friction/armature + actuator
        # gains + cube mass/material + gravity). Parity with PickPlace +
        # Stack — the docs claim all three built-in tasks have DR enabled
        # by default.
        from openso101.sim2real.domain_randomization.physics import attach_all_physics_dr
        attach_all_physics_dr(
            self.events,
            robot_asset_name="robot",
            object_asset_names=("object",),
            include_gravity=True,
        )

    # ---------- variant hooks ----------

    def configure_play(self, enabled: bool) -> None:
        """Eval variant: 50 envs, no observation corruption."""
        if not enabled:
            return
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

    def configure_visual_dr(self, enabled: bool = True) -> None:
        """Attach dome-light + object-color randomization at episode reset."""
        if not enabled:
            return
        from openso101.sim2real.domain_randomization.visual import attach_visual_dr
        attach_visual_dr(self.events, object_asset_name="object")

    def configure_action_mode(self, mode: str) -> None:
        """`'rl'` keeps the trained-policy action setup; `'teleop'` swaps to
        absolute joint positions on the leader-arm SO_ARM101_TELEOP_CFG and
        drops RL-only managers (rewards/terminations/curriculum)."""
        if mode == "rl":
            return
        if mode == "teleop":
            self.actions = TeleopActionsCfg()
            _configure_so101_lift_scene(self, robot_cfg=SO_ARM101_TELEOP_CFG)
            # Restore body_name for the command, otherwise the manager errors.
            self.commands.object_pose.body_name = ["gripper"]
            # Strip jaw contact sensors: the teleop robot has
            # activate_contact_sensors=False (no contact reporter API on its
            # bodies), and no teleop reward consumes the signal.
            self.scene.gripper_jaw_contact = None
            self.scene.moving_jaw_contact = None
            self.rewards = None
            self.terminations = None
            self.curriculum = None
            # Hide RL-only debug markers (goal sphere + EE axis triad) so the
            # IL recording cameras see a clean scene.
            self.commands.object_pose.debug_vis = False
            self.scene.ee_frame.debug_vis = False
            self.episode_length_s = 3600.0
            self.scene.num_envs = 1
            self.scene.env_spacing = 2.5
            self.observations.policy.enable_corruption = False
            self.decimation = 2
            self.sim.dt = 1 / 120
            self.sim.render_interval = self.decimation
            return
        raise UnsupportedVariantError(
            f"action_mode={mode!r} not supported; expected 'rl' or 'teleop'."
        )
