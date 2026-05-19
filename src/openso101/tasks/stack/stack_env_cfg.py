# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Stack task -- stack one cube on top of another.

Single-class env cfg with variant hooks. Collapses legacy `StackEnvCfg`,
`SoArm101StackCubeEnvCfg`, `_PLAY`, `WithCameras`, and teleop variants from
`safe_sim2real.tasks.composite.stack.{stack_env_cfg, joint_pos_env_cfg}` into
a single `OpenSO101EnvCfg` subclass whose `configure_*` hooks mutate the
cfg in-place per `gym.make()` kwarg.

The base RL configuration also attaches physics domain randomization via
:func:`openso101.sim2real.domain_randomization.physics.attach_all_physics_dr`.
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
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import openso101.tasks.stack.mdp as mdp
from openso101.envs import OpenSO101EnvCfg, TeleopActionsCfg, UnsupportedVariantError
from openso101.robots import (
    SO101_ARM_JOINT_NAMES,
    SO101_GRIPPER_CLOSED_POS,
    SO101_GRIPPER_JOINT_NAME,
    SO101_GRIPPER_JOINT_NAMES,
    SO101_GRIPPER_OPEN_POS,
)
from openso101.robots.so101.so_arm101 import SO_ARM101_CFG, SO_ARM101_TELEOP_CFG
from openso101.sim2real.domain_randomization.physics import attach_all_physics_dr
from openso101.tasks.shared.objects import SO101_CUBE_SIZE, so101_cube_object_cfg
from openso101.tasks.shared.rl_defaults import (
    SO101_ACTION_RATE_WEIGHT,
    SO101_CONTROLLED_OBJECT_MIN_HEIGHT,
    SO101_GOAL_TRACKING_FINE_STD,
    SO101_JOINT_POS_DELTA_WEIGHT,
    SO101_JOINT_VEL_WEIGHT,
    SO101_REACH_REWARD_STD,
    SO101_SMOOTHNESS_CURRICULUM_STEPS,
    SO101_SMOOTHNESS_CURRICULUM_WEIGHT,
)

##
# Scene definition
##

# Cube size used throughout (edge length in metres)
CUBE_SIZE = SO101_CUBE_SIZE


@configclass
class StackSceneCfg(InteractiveSceneCfg):
    """Scene with a robot arm and two coloured cubes to stack."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    cube_top: RigidObjectCfg = MISSING
    cube_bottom: RigidObjectCfg = MISSING

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
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_top_pos = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("cube_top")},
        )
        cube_bottom_pos = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("cube_bottom")},
        )
        cube_top_to_goal = ObsTerm(
            func=mdp.cube_top_to_cube_bottom_offset,
            params={"cube_height": CUBE_SIZE},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # No ObsTerm in this group sets a noise=Unoise/Gnoise, so observation
            # corruption is currently a no-op. Set False to avoid the misleading
            # "I think I'm noisy" state; re-enable only when per-term noise is
            # intentionally configured.
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_cube_bottom = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.15), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_bottom", body_names=".*"),
        },
    )
    reset_cube_top = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.15), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_top", body_names=".*"),
        },
    )


@configclass
class RewardsCfg:
    reaching_cube_top = RewTerm(
        func=mdp.ee_to_cube_top_distance, params={"std": SO101_REACH_REWARD_STD}, weight=1.0,
    )
    lifting_cube_top = RewTerm(
        func=mdp.cube_top_is_lifted,
        params={"minimal_height": SO101_CONTROLLED_OBJECT_MIN_HEIGHT},
        weight=10.0,
    )
    aligning_cubes = RewTerm(
        func=mdp.cubes_xy_aligned, params={"std": 0.08}, weight=8.0,
    )
    stacking_cubes = RewTerm(
        func=mdp.cubes_stacked, params={"std": SO101_GOAL_TRACKING_FINE_STD, "cube_height": CUBE_SIZE}, weight=20.0,
    )
    stacking_cubes_fine = RewTerm(
        func=mdp.cubes_stacked, params={"std": 0.015, "cube_height": CUBE_SIZE}, weight=5.0,
    )
    released_on_stack = RewTerm(
        func=mdp.cube_released_on_stack,
        params={
            "xy_threshold": 0.02,
            "height_tolerance": 0.012,
            "cube_height": CUBE_SIZE,
            "open_threshold": 0.35,
            "object_static_threshold": 0.2,
            "gripper_cfg": SceneEntityCfg("robot", joint_names=list(SO101_GRIPPER_JOINT_NAMES)),
        },
        weight=12.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=SO101_ACTION_RATE_WEIGHT)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=SO101_JOINT_VEL_WEIGHT, params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_pos_delta = RewTerm(
        func=mdp.joint_pos_delta_l2, weight=SO101_JOINT_POS_DELTA_WEIGHT, params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_top_dropped = DoneTerm(
        func=mdp.cube_dropped, params={"minimum_height": -0.05, "cube_cfg": SceneEntityCfg("cube_top")},
    )
    cube_bottom_dropped = DoneTerm(
        func=mdp.cube_dropped, params={"minimum_height": -0.05, "cube_cfg": SceneEntityCfg("cube_bottom")},
    )
    stacked_success = DoneTerm(
        func=mdp.cubes_stacked_success,
        params={
            "xy_threshold": 0.015,
            "height_tolerance": 0.01,
            "cube_height": CUBE_SIZE,
            "open_threshold": 0.35,
            "object_static_threshold": 0.2,
            "gripper_cfg": SceneEntityCfg("robot", joint_names=list(SO101_GRIPPER_JOINT_NAMES)),
        },
    )


@configclass
class CurriculumCfg:
    """Smoothness ramp after early stack behavior can emerge.

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


def _red_cube_cfg(prim_path: str, init_pos: list[float]):
    """Create a red (bottom) cube rigid object config."""
    return so101_cube_object_cfg(
        prim_path=prim_path,
        init_pos=init_pos,
        diffuse_color=(0.9, 0.1, 0.1),
        size=CUBE_SIZE,
    )


def _blue_cube_cfg(prim_path: str, init_pos: list[float]):
    """Create a blue (top) cube rigid object config."""
    return so101_cube_object_cfg(
        prim_path=prim_path,
        init_pos=init_pos,
        diffuse_color=(0.1, 0.3, 0.9),
        size=CUBE_SIZE,
    )


def _configure_so101_stack_scene(cfg: "StackEnvCfg", robot_cfg=None) -> None:
    """Populate the SO-101 robot, two cubes, and EE frame on a Stack scene."""
    if robot_cfg is None:
        robot_cfg = SO_ARM101_CFG
    cfg.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cfg.scene.cube_bottom = _red_cube_cfg(
        prim_path="{ENV_REGEX_NS}/CubeBottom",
        init_pos=[0.35, 0.05, CUBE_SIZE / 2],
    )
    cfg.scene.cube_top = _blue_cube_cfg(
        prim_path="{ENV_REGEX_NS}/CubeTop",
        init_pos=[0.35, -0.1, CUBE_SIZE / 2],
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
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
        ],
    )


@configclass
class StackEnvCfg(OpenSO101EnvCfg):
    """Single-class Stack cfg with variant hooks."""

    scene: StackSceneCfg = StackSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # Base RL setup (from StackEnvCfg + SoArm101StackCubeEnvCfg).
        self.decimation = 2
        self.episode_length_s = 10.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        # See lift_env_cfg for rationale: Lior-style 32-iteration solver
        # overflows the default ~64 MB GPU contact buffer at high num_envs.
        # Stack has two cubes which doubles the contact pair count.
        self.sim.physx.gpu_collision_stack_size = 512 * 1024 * 1024

        # SO-101 scene wiring.
        _configure_so101_stack_scene(self)

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

        # Phase 1 - physics domain randomization (ranges live in
        # openso101.sim2real.domain_randomization.physics).
        attach_all_physics_dr(
            self.events,
            robot_asset_name="robot",
            object_asset_names=("cube_top", "cube_bottom"),
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
        """Attach dome-light + per-cube color randomization at reset.

        Stack has two distinguishable cubes (bottom + top) so each gets
        its own independent recolor — otherwise the policy could exploit
        always-paired colors instead of learning to discriminate by
        position alone.
        """
        if not enabled:
            return
        from openso101.sim2real.domain_randomization.visual import attach_visual_dr
        attach_visual_dr(self.events, object_asset_name=("cube_bottom", "cube_top"))

    def configure_action_mode(self, mode: str) -> None:
        """`'rl'` keeps the trained-policy action setup; `'teleop'` swaps to
        absolute joint positions on the leader-arm SO_ARM101_TELEOP_CFG and
        drops RL-only managers (rewards/terminations/curriculum)."""
        if mode == "rl":
            return
        if mode == "teleop":
            self.actions = TeleopActionsCfg()
            _configure_so101_stack_scene(self, robot_cfg=SO_ARM101_TELEOP_CFG)
            self.rewards = None
            self.terminations = None
            self.curriculum = None
            # Hide the EE axis triad so the IL recording cameras see a
            # clean scene. (Stack has no goal-pose command, only an EE
            # FrameTransformer to hide.)
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
