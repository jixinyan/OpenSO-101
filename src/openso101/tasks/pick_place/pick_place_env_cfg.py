# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Pick-and-place env -- 3-stage curriculum (lift -> carry -> place).

This task is **distinct** from the atomic Lift skill, which uses Isaac Lab's
official lift reward design with a single random air-pose goal. Pick-and-place
here is a true sequence: the cube must be lifted, carried laterally, and
placed back on the table.

The mechanism is "curriculum via goal placement". One green ball is shown in
the scene; its location is chosen by a custom :class:`CurriculumGoalCommand`
based on a per-env stage:

- Stage 0: ball above cube spawn xy at z = 0.10  (teaches lifting)
- Stage 1: ball above final place xy at z = 0.15 (teaches carrying)
- Stage 2: ball on table at the same final xy     (teaches placing)

The cube touching a stage's ball advances the stage; only stage 2's success
terminates the episode. The reward layout intentionally stays close to atomic
Lift: reach, lift, staged object-goal tracking, sparse stage completion, and
smoothness. The goal command moves the green ball after each stage transition;
the reward terms stay split by stage so the task remains diagnosable and each
phase can use the right gate.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
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

import openso101.tasks.pick_place.mdp as mdp
from openso101.envs import OpenSO101EnvCfg, TeleopActionsCfg, UnsupportedVariantError
from openso101.robots import (
    SO101_ARM_JOINT_NAMES,
    SO101_GRIPPER_CLOSED_POS,
    SO101_GRIPPER_JOINT_NAME,
    SO101_GRIPPER_JOINT_NAMES,
    SO101_GRIPPER_OPEN_POS,
)
from openso101.robots.so101.so_arm101 import (
    SO_ARM101_CFG,
    SO_ARM101_TELEOP_CFG,
    SO101_USD_TABLETOP_ROOT_Z,
)
from openso101.tasks.shared.objects import so101_cube_object_cfg
from openso101.tasks.shared.rl_defaults import (
    SO101_ACTION_RATE_WEIGHT,
    SO101_GOAL_TRACKING_STD,
    SO101_JOINT_VEL_WEIGHT,
    SO101_REACH_REWARD_COARSE_STD,
    SO101_REACH_REWARD_COARSE_WEIGHT,
    SO101_REACH_REWARD_STD,
    SO101_SMOOTHNESS_CURRICULUM_STEPS,
    SO101_SMOOTHNESS_CURRICULUM_WEIGHT,
)

SO101_USD_TABLETOP_BASE_OFFSET = -SO101_USD_TABLETOP_ROOT_Z

_GOAL_SPHERE_RADIUS = 0.03
_CUBE_CONTACT_RADIUS = 0.015
_SUCCESS_REWARD = 25.0

# Green goal sphere marker, with one mesh per stage so the command can still
# index by stage. The geometry/color stays the same; only the position changes.
CURRICULUM_GOAL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Curriculum/goal",
    markers={
        "stage_0_lift": sim_utils.SphereCfg(
            radius=_GOAL_SPHERE_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.9, 0.1), opacity=0.5,
            ),
        ),
        "stage_1_carry": sim_utils.SphereCfg(
            radius=_GOAL_SPHERE_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.9, 0.1), opacity=0.5,
            ),
        ),
        "stage_2_place": sim_utils.SphereCfg(
            radius=_GOAL_SPHERE_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.9, 0.1), opacity=0.5,
            ),
        ),
    },
)


##
# Scene
##


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Scene: robot + cube on table."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg = MISSING

    # Contact sensors on the two opposing jaw bodies, filtered to the cube prim.
    # Used by the grasp-confirmation reward gate; force_matrix_w fires only
    # when the corresponding jaw is pressing against the cube. Set to None in
    # teleop mode (no rewards consume them, and the teleop robot config does
    # not enable contact reporting on its bodies).
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
    fixed_fingertip_pad: AssetBaseCfg | None = None
    moving_fingertip_pad: AssetBaseCfg | None = None

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """3-stage curriculum goal (lift -> carry -> place)."""

    object_pose = mdp.CurriculumGoalCommandCfg(
        asset_name="robot",
        object_name="object",
        # `resampling_time_range` set to a value larger than any episode so the
        # curriculum command's `_resample_command` only fires at episode reset.
        # Stage transitions happen via `_update_command`, not by resampling.
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        lift_height=0.10,
        carry_height=0.15,
        place_goal=(0.20, 0.18, 0.02),
        advance_threshold=_GOAL_SPHERE_RADIUS,
        object_contact_radius=_CUBE_CONTACT_RADIUS,
        goal_pose_visualizer_cfg=CURRICULUM_GOAL_MARKER_CFG,
    )


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Standard proprioception + cube + curriculum-goal observation.

    `target_object_position` reads the curriculum command's goal directly via
    `mdp.generated_commands`, so the policy sees whichever stage's goal is
    currently active for its env.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # No ObsTerm in this group sets noise=..., so corruption is
            # currently a no-op. Re-enable only with explicit per-term noise.
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Cube reset to a random spot on the table."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    """Upstream-pattern stage chain with height-only gating (not grasp-AND).

    The 3-stage curriculum command (lift/carry/place) advances on cube-goal
    proximity, not on grasp state. Stage rewards gate on object height >=
    minimal_height only; the policy must lift the cube to unlock stage_0/1
    rewards, and the cube must be on the table (no height gate) for stage_2.

    The 'grasped' contact-sensor reward survives as a small dense bonus,
    but is no longer a multiplicative gate on anything.
    """

    # Two-scale reach: coarse covers ~35cm starting distance, fine sharpens at grasp range.
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

    # Bonus only -- NOT a gate. Rewards confirmed contact-pinch (the only
    # signal that distinguishes "jaws closed in air" from "jaws closed on
    # cube" before the cube lifts).
    grasped = RewTerm(
        func=mdp.grasped_reward,
        params={"force_threshold": 0.5},
        weight=3.0,
    )

    # Height-only lift (no contact gate).
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04},
        weight=15.0,
    )

    stage_0_lift_goal = RewTerm(
        func=mdp.cube_to_curriculum_stage_goal_height_gated,
        params={"stage": 0, "std": SO101_GOAL_TRACKING_STD, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=12.0,
    )

    stage_0_complete_bonus = RewTerm(
        func=mdp.stage_completion_bonus,
        params={"completed_stage": 0, "command_name": "object_pose"},
        weight=10.0,
    )

    stage_1_carry_goal = RewTerm(
        func=mdp.cube_to_curriculum_stage_goal_height_gated,
        params={"stage": 1, "std": SO101_GOAL_TRACKING_STD, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    stage_1_complete_bonus = RewTerm(
        func=mdp.stage_completion_bonus,
        params={"completed_stage": 1, "command_name": "object_pose"},
        weight=10.0,
    )

    # Stage 2: place -- NO height gate, since the policy must release the cube.
    stage_2_place_goal = RewTerm(
        func=mdp.cube_to_curriculum_stage_goal,
        params={"stage": 2, "std": SO101_GOAL_TRACKING_STD, "minimal_height": None, "command_name": "object_pose"},
        weight=16.0,
    )

    success_bonus = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": ["success"]},
        weight=_SUCCESS_REWARD,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=SO101_ACTION_RATE_WEIGHT)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=SO101_JOINT_VEL_WEIGHT,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Episode ends only on time-out, drop, or stage-2 success.

    Crucially, advancing through stages 0 and 1 does NOT end the episode --
    it just moves the goal sphere. Only entering the stage-2 (place) sphere
    counts as task completion.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    success = DoneTerm(
        func=mdp.curriculum_complete,
        params={"command_name": "object_pose", "threshold": _GOAL_SPHERE_RADIUS},
    )


@configclass
class CurriculumCfg:
    """Gentle smoothness ramp after early lift/carry behavior can emerge."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "action_rate",
            "weight": SO101_SMOOTHNESS_CURRICULUM_WEIGHT,
            "num_steps": SO101_SMOOTHNESS_CURRICULUM_STEPS,
        },
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "joint_vel",
            "weight": SO101_SMOOTHNESS_CURRICULUM_WEIGHT,
            "num_steps": SO101_SMOOTHNESS_CURRICULUM_STEPS,
        },
    )


##
# Environment configuration
##


def _configure_so101_pick_place_scene(cfg: "PickPlaceEnvCfg", robot_cfg=None) -> None:
    """Populate the SO-101 robot, cube, and EE frame on a PickPlace scene."""
    if robot_cfg is None:
        robot_cfg = SO_ARM101_CFG
    cfg.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.object = so101_cube_object_cfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_pos=[0.3, 0.0, 0.015],
        diffuse_color=(0.1, 0.8, 0.2),
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
class PickPlaceEnvCfg(OpenSO101EnvCfg):
    """Single-class PickPlace cfg with variant hooks."""

    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # Base RL setup (from PickPlaceEnvCfg + SoArm101PickPlaceEnvCfg).
        self.decimation = 2
        # Episodes are longer than the lift task because the policy must complete
        # all three stages back-to-back. 8s ~= 400 steps at 50 Hz control.
        self.episode_length_s = 8.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        # See lift_env_cfg for rationale: Lior-style 32-iteration solver
        # overflows the default ~64 MB GPU contact buffer at high num_envs.
        self.sim.physx.gpu_collision_stack_size = 512 * 1024 * 1024

        # SO-101 scene wiring.
        _configure_so101_pick_place_scene(self)

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

        # Adjust curriculum goal heights to compensate for the canonical SO101 USD
        # being placed at z=SO101_USD_TABLETOP_ROOT_Z (table-top mounted layout).
        self.commands.object_pose.lift_height += SO101_USD_TABLETOP_BASE_OFFSET
        self.commands.object_pose.carry_height += SO101_USD_TABLETOP_BASE_OFFSET
        place_x, place_y, place_z = self.commands.object_pose.place_goal
        self.commands.object_pose.place_goal = (
            place_x,
            place_y,
            place_z + SO101_USD_TABLETOP_BASE_OFFSET,
        )
        self.rewards.stage_0_lift_goal.params["minimal_height"] += SO101_USD_TABLETOP_BASE_OFFSET
        self.rewards.stage_1_carry_goal.params["minimal_height"] += SO101_USD_TABLETOP_BASE_OFFSET

        # Physics DR (robot link mass + joint friction/armature + actuator
        # gains + cube mass/material + gravity). Same call Stack already
        # makes — having parity across all three RL tasks is the documented
        # sim2real default in docs/concepts/sim2real.md.
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
        """Attach visual DR (light intensity/color + object color) at reset.

        Cheap perception sim2real win — defaults match the leisaac +
        lerobot_so101_teleop recommendations. Pass ``enabled=False`` to
        ablate, which is useful when comparing trained-policy transfer
        with vs. without visual DR.
        """
        if not enabled:
            return
        from openso101.sim2real.domain_randomization.visual import attach_visual_dr
        attach_visual_dr(self.events, object_asset_name="object")

    def configure_action_mode(self, mode: str) -> None:
        """`'rl'` keeps the trained-policy action setup; `'teleop'` swaps to
        absolute joint positions on the leader-arm SO_ARM101_TELEOP_CFG, drops
        RL-only managers (rewards/terminations/curriculum), and uses long
        episodes + a single env."""
        if mode == "rl":
            return
        if mode == "teleop":
            self.actions = TeleopActionsCfg()
            # Re-spawn the scene with the teleop robot articulation.
            _configure_so101_pick_place_scene(self, robot_cfg=SO_ARM101_TELEOP_CFG)
            # Strip jaw contact sensors: teleop robot config matches Lior
            # (activate_contact_sensors=False), so the bodies have no contact
            # reporter API, and no teleop reward consumes the signal anyway.
            self.scene.gripper_jaw_contact = None
            self.scene.moving_jaw_contact = None
            self.rewards = None
            self.terminations = None
            self.curriculum = None
            # Hide RL-only debug markers so the IL recording cameras see a
            # clean scene: the green goal sphere lives at
            # /Visuals/Curriculum/goal and the EE coloured-axis triad lives
            # at /Visuals/FrameTransformer. Both are training aids and would
            # leak into the wrist + overhead camera observations otherwise.
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
