# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

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
    SO101_JOINT_VEL_WEIGHT,
    SO101_PICK_CARRY_COEFF,
    SO101_PICK_GOAL_BONUS,
    SO101_PICK_GRASP_HOLD_WEIGHT,
    SO101_PICK_GRASP_ONSET_BONUS,
    SO101_PICK_PREGRASP_COEFF,
    SO101_SMOOTHNESS_CURRICULUM_STEPS,
    SO101_SMOOTHNESS_CURRICULUM_WEIGHT,
)

SO101_USD_TABLETOP_BASE_OFFSET = -SO101_USD_TABLETOP_ROOT_Z

_GOAL_SPHERE_RADIUS = 0.03
_CUBE_CONTACT_RADIUS = 0.015

# Green goal sphere marker. The command's visualizer selects a mesh by integer
# `marker_indices`, so three identical entries are kept in order to cover every
# index the command may emit; the geometry/color is the same and only the
# position changes. The key strings are cosmetic and carry no curriculum meaning.
CURRICULUM_GOAL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Curriculum/goal",
    markers={
        "goal_a": sim_utils.SphereCfg(
            radius=_GOAL_SPHERE_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.9, 0.1), opacity=0.5,
            ),
        ),
        "goal_b": sim_utils.SphereCfg(
            radius=_GOAL_SPHERE_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.9, 0.1), opacity=0.5,
            ),
        ),
        "goal_c": sim_utils.SphereCfg(
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
    """抓取后抬升物体，搬运至目标上方，随后放置和释放。"""

    object_pose = mdp.CurriculumGoalCommandCfg(
        asset_name="robot",
        object_name="object",
        # 目标重置仅在 episode 开始时执行。
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        lock_stage=None,
        lift_height=0.10,
        carry_height=0.15,
        place_goal=(0.24, -0.15, 0.015),
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
        # Contact-confirmed grasp state ([N,1] float, 1.0 when both jaws pinch
        # the cube). Gives the policy the same grasp signal the reward stack
        # gates on, so it doesn't have to infer "am I holding it?" from proprio.
        grasp_state = ObsTerm(func=mdp.object_grasped_obs, params={"force_threshold": 0.5})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # Observation DR (joint-pos/vel noise) is attached in the env cfg
            # __post_init__ and corruption is enabled there; the per-term noise
            # is sampled by the observation manager when enable_corruption=True.
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
            # Cube reset jitter around init_pos=[0.3, 0.0, 0.015]. Widened from
            # (±0.03, ±0.025) -> (±0.05, ±0.04) so the variation is visually
            # obvious in the teleop viewport (small shifts of 1-2 cm were below
            # the operator's perceptual threshold and felt like "no random").
            # Max radial reach √((0.30+0.05)² + 0.04²) ≈ 0.352 m — slightly
            # past the SO-101's ~0.30 m comfortable zone at the corner case
            # but still reachable; teleop will warn the operator if a target
            # is unreachable.
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.04, 0.04), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    """Sentinel-style delta-distance shaping (pregrasp / hold / carry / goal).

    Each term's weight IS the shaping coefficient; the RewardManager applies
    ``weight * dt``. Delta shaping rewards *progress* (distance reduction), so
    hovering near a target pays nothing — closing the "touch-and-farm" exploit
    that absolute tanh-distance rewards permit.
    """

    # Approach: reward reducing eef->object distance while NOT grasping. The
    # mode flips off (silent) the moment a contact-confirmed grasp is achieved.
    pregrasp_approach = RewTerm(
        func=mdp.pregrasp_approach_shaping,
        params={"force_threshold": 0.5},
        weight=SO101_PICK_PREGRASP_COEFF,
    )

    # Hold: per-step reward for a contact-confirmed grasp (both jaws pinching
    # the cube). A weak steady-state signal once contact is made; the discrete
    # credit for *achieving* the grasp lives in ``grasp_onset`` below, and the
    # actual delivery driver is ``carry_to_goal`` (see rl_defaults audit note).
    grasp_hold = RewTerm(
        func=mdp.grasped_reward,
        params={"force_threshold": 0.5},
        weight=SO101_PICK_GRASP_HOLD_WEIGHT,
    )

    # Grasp onset: one-shot reward on the single step the grasp first confirms
    # (False->True transition). Sharpens the reach->grasp boundary that the weak
    # steady-state grasp_hold alone under-rewards. (Audit hypothesis weight;
    # needs a training run to validate — see rl_defaults.)
    grasp_onset = RewTerm(
        func=mdp.grasp_onset_bonus,
        params={"force_threshold": 0.5},
        weight=SO101_PICK_GRASP_ONSET_BONUS,
    )

    # 搬运奖励根据当前阶段的目标距离变化计算。
    carry_to_goal = RewTerm(
        func=mdp.carry_to_goal_shaping,
        params={"command_name": "object_pose", "force_threshold": 0.5},
        weight=SO101_PICK_CARRY_COEFF,
    )

    # 物体释放后在放置区域保持稳定，获得完成奖励。
    success_bonus = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": ["success"]},
        weight=SO101_PICK_GOAL_BONUS,
    )

    # Smoothness (physical-sanity infra, orthogonal to the task shaping above):
    # joint_vel is active from step 0 to stop the arm flailing the cube off the
    # table; action_rate is curriculum-ramped so it doesn't suppress early
    # exploration.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=SO101_ACTION_RATE_WEIGHT)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=SO101_JOINT_VEL_WEIGHT,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """检查时间限制、物体越过桌面下方和完整放置成功。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    success = DoneTerm(
        func=mdp.released_at_place_goal,
        params={
            "command_name": "object_pose",
            "settle_seconds": 0.5,
        },
    )


@configclass
class CurriculumCfg:
    """Gentle smoothness ramp after early lift/carry behavior can emerge.

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

        # 根据 SO-101 基座高度计算 robot frame 中的目标高度。
        self.commands.object_pose.lift_height += SO101_USD_TABLETOP_BASE_OFFSET
        self.commands.object_pose.carry_height += SO101_USD_TABLETOP_BASE_OFFSET
        place_x, place_y, place_z = self.commands.object_pose.place_goal
        self.commands.object_pose.place_goal = (
            place_x,
            place_y,
            place_z + SO101_USD_TABLETOP_BASE_OFFSET,
        )

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

        # Observation DR: per-step uniform noise on joint_pos/joint_vel obs
        # terms (Feetech encoder noise model). enable_corruption=True so the
        # observation manager actually samples and applies the noise — without
        # it the per-term noise cfgs are inert. configure_play() turns this back
        # off for clean eval.
        from openso101.sim2real.domain_randomization.observation import attach_observation_dr
        attach_observation_dr(self.observations.policy)
        self.observations.policy.enable_corruption = True

    # ---------- variant hooks ----------

    def configure_play(self, enabled: bool) -> None:
        """Eval variant: 50 envs, no observation corruption."""
        if not enabled:
            return
        self.scene.num_envs = 50
        self.action_dr_enabled = False
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
            self.action_dr_enabled = False
            self.actions = TeleopActionsCfg()
            # Re-spawn the scene with the teleop robot articulation.
            _configure_so101_pick_place_scene(self, robot_cfg=SO_ARM101_TELEOP_CFG)
            # Strip jaw contact sensors: the teleop robot has
            # activate_contact_sensors=False (no contact reporter API on its
            # bodies), and no teleop reward consumes the signal.
            self.scene.gripper_jaw_contact = None
            self.scene.moving_jaw_contact = None
            # The grasp_state obs term reads those (now-absent) contact sensors,
            # so drop it for teleop. None terms are skipped by the obs manager.
            self.observations.policy.grasp_state = None
            self.rewards = None
            self.terminations = None
            self.curriculum = None
            # Lock the curriculum goal at the final place sphere — operators
            # want one explicit "place here" target, not the RL lift / carry /
            # place chain. With stage pinned at 2, _update_command's
            # `stage < 2` advancement gate naturally skips. The on-table place
            # sphere remains visible (cfg-level debug_vis stays True so
            # programmatic users get the marker; `_cmd_record`'s --goal-region
            # default also lights it up for CLI users).
            self.commands.object_pose.lock_stage = 2
            # Hide the EE axis triad so the IL recording cameras see a clean scene.
            self.scene.ee_frame.debug_vis = False
            self.episode_length_s = 3600.0
            self.scene.num_envs = 1
            self.scene.env_spacing = 2.5
            self.observations.policy.enable_corruption = False
            self.decimation = 2
            self.sim.dt = 1 / 120
            self.sim.render_interval = self.decimation
            # __post_init__ sized PhysX GPU buffers for RL training at
            # num_envs=4096. Teleop uses num_envs=1, so revert to PhysX
            # defaults to free up ~400+ MB of VRAM for IL play's policy
            # inference (especially diffusion, which is ~1 GB + denoising
            # intermediates that push the 8 GB laptop GPU into OOM).
            self.sim.physx.gpu_collision_stack_size = 64 * 1024 * 1024  # default 64 MB
            self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024
            self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024
            return
        raise UnsupportedVariantError(
            f"action_mode={mode!r} not supported; expected 'rl' or 'teleop'."
        )
