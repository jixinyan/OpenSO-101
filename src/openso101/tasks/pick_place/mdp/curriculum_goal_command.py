# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""3-stage curriculum goal command for pick-and-place.

The policy always solves the same task: "get the cube to the green ball".
The curriculum is in *where the ball is*, not in the reward function:

- Stage 0 (Lift): goal is directly above the cube's spawn xy at a fixed height.
  Touching it teaches the policy to grasp + lift cleanly.
- Stage 1 (Carry): goal jumps above the final placement xy at a fixed carry
  height. Touching it teaches the policy to transport the lifted cube laterally
  without dropping.
- Stage 2 (Place): goal moves down to the table surface at the same placement xy.
  Touching it teaches the policy to lower and release.

Stage advancement happens *within* an episode: as soon as the cube touches the
current goal sphere, that env's stage increments and the marker jumps to the
next stage's location. The policy must then track the goal again. The episode
terminates only at stage-2 success (or time-out / cube drop).

The marker config provides one sphere mesh per stage so the visualizer can
select by stage. The task config can keep them visually identical if the only
intended cue is the goal position moving.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CurriculumGoalCommand(CommandTerm):
    """Per-env stage tracker that exposes `goal - in robot root frame` as the command.

    Subclasses :class:`isaaclab.managers.CommandTerm` so it integrates seamlessly
    with the rest of Isaac Lab (reward functions and observations read it via
    ``env.command_manager.get_command(name)``; debug-vis spawns the marker
    automatically).
    """

    cfg: "CurriculumGoalCommandCfg"

    def __init__(self, cfg: "CurriculumGoalCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]

        n = self.num_envs
        # Per-env state.
        self.stage = torch.zeros(n, dtype=torch.long, device=self.device)
        self.cube_spawn_xy_b = torch.zeros(n, 2, device=self.device)
        # Goal in robot root frame (the "command" returned to consumers).
        self.goal_pos_b = torch.zeros(n, 3, device=self.device)
        # Goal in world frame (used only for marker visualization).
        self.goal_pos_w = torch.zeros(n, 3, device=self.device)

        # Cache fixed-stage goals as tensors. Stage 1 derives its x/y from the
        # final place goal so the carry target is always directly above the
        # eventual placement location.
        self._place_goal_b = torch.tensor(cfg.place_goal, device=self.device)
        self._carry_goal_b = self._place_goal_b.clone()
        self._carry_goal_b[2] = cfg.carry_height

        # Metrics tracked per step.
        self.just_completed_stage = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.metrics["stage"] = torch.zeros(n, device=self.device)
        self.metrics["distance_to_goal"] = torch.zeros(n, device=self.device)
        self.metrics["surface_distance_to_goal"] = torch.zeros(n, device=self.device)

    def __str__(self) -> str:
        return (
            f"CurriculumGoalCommand(num_envs={self.num_envs}, "
            f"lift_height={self.cfg.lift_height}, "
            f"carry_height={self.cfg.carry_height}, place={self.cfg.place_goal}, "
            f"advance_threshold={self.cfg.advance_threshold}, "
            f"object_contact_radius={self.cfg.object_contact_radius})"
        )

    @property
    def command(self) -> torch.Tensor:
        """Goal position in robot root frame, shape (num_envs, 3)."""
        return self.goal_pos_b

    def goal_for_stage(self, stage: int) -> torch.Tensor:
        """Return the per-env goal position for a specific curriculum stage."""
        if stage == 0:
            goal = torch.zeros_like(self.goal_pos_b)
            goal[:, 0] = self.cube_spawn_xy_b[:, 0]
            goal[:, 1] = self.cube_spawn_xy_b[:, 1]
            goal[:, 2] = self.cfg.lift_height
            return goal
        if stage == 1:
            return self._carry_goal_b.expand_as(self.goal_pos_b)
        if stage == 2:
            return self._place_goal_b.expand_as(self.goal_pos_b)
        raise ValueError(f"Invalid curriculum stage: {stage}")

    def is_touching_goal(
        self,
        cube_pos_b: torch.Tensor,
        goal_pos_b: torch.Tensor | None = None,
        threshold: float | None = None,
    ) -> torch.Tensor:
        """Return true when the cube surface touches the goal sphere.

        The marker radius describes the visible goal sphere. The cube is not a
        point, so contact should be measured with an expanded threshold:
        ``sphere_radius + object_contact_radius``.
        """
        goal = self.goal_pos_b if goal_pos_b is None else goal_pos_b
        sphere_radius = self.cfg.advance_threshold if threshold is None else threshold
        contact_threshold = sphere_radius + self.cfg.object_contact_radius
        distance = torch.norm(cube_pos_b - goal, dim=1)
        return distance <= contact_threshold

    # --- Implementation hooks called by CommandTerm ----------------------

    def _update_metrics(self):
        """Per-step metrics. Called before `_update_command` each step."""
        cube_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.object.data.root_pos_w,
        )
        distance = torch.norm(cube_pos_b - self.goal_pos_b, dim=1)
        self.metrics["distance_to_goal"] = distance
        self.metrics["surface_distance_to_goal"] = torch.clamp(
            distance - (self.cfg.advance_threshold + self.cfg.object_contact_radius),
            min=0.0,
        )
        self.metrics["stage"] = self.stage.float()

    def _resample_command(self, env_ids: Sequence[int]):
        """Episode reset for these envs: stage <- 0 (or lock_stage), record
        cube spawn xy, set goal.

        When ``cfg.lock_stage`` is set (teleop uses ``lock_stage=2`` to show
        only the final on-table place sphere), the per-env stage tensor is
        pinned to that value at reset. ``_update_command``'s advancement
        gate ``stage < 2`` then naturally skips, so no further changes are
        needed to disable stage chaining.
        """
        start_stage = 0 if self.cfg.lock_stage is None else int(self.cfg.lock_stage)
        self.stage[env_ids] = start_stage
        self.just_completed_stage[env_ids] = -1
        cube_pos_w = self.object.data.root_pos_w[env_ids]
        cube_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            cube_pos_w,
        )
        self.cube_spawn_xy_b[env_ids] = cube_pos_b[:, :2]
        self._refresh_goals(torch.as_tensor(env_ids, device=self.device, dtype=torch.long))

    def _update_command(self):
        """Each step: advance stage for envs whose cube reached the current goal."""
        self.just_completed_stage.fill_(-1)
        cube_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.object.data.root_pos_w,
        )
        # Advance only envs that are not yet at the final stage.
        reached = self.is_touching_goal(cube_pos_b) & (self.stage < 2)
        if reached.any():
            advance_ids = reached.nonzero(as_tuple=False).flatten()
            self.just_completed_stage[advance_ids] = self.stage[advance_ids]
            self.stage[advance_ids] += 1
            self._refresh_goals(advance_ids)

        # Refresh world-frame goal each step (robot root may move; for fixed-base it's constant).
        self.goal_pos_w, _ = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.goal_pos_b,
        )

    # --- Stage <-> goal mapping ------------------------------------------

    def _refresh_goals(self, env_ids: torch.Tensor):
        """Vectorised: set goal_pos_b for the given envs based on their stage."""
        stages = self.stage[env_ids]

        lift_mask = stages == 0
        if lift_mask.any():
            ids = env_ids[lift_mask]
            self.goal_pos_b[ids, 0] = self.cube_spawn_xy_b[ids, 0]
            self.goal_pos_b[ids, 1] = self.cube_spawn_xy_b[ids, 1]
            self.goal_pos_b[ids, 2] = self.cfg.lift_height

        carry_mask = stages == 1
        if carry_mask.any():
            ids = env_ids[carry_mask]
            self.goal_pos_b[ids] = self._carry_goal_b

        place_mask = stages == 2
        if place_mask.any():
            ids = env_ids[place_mask]
            self.goal_pos_b[ids] = self._place_goal_b

    # --- Visualization ----------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # `marker_indices` selects which mesh from the cfg's `markers` dict to render
        # for each env.
        self.goal_visualizer.visualize(
            translations=self.goal_pos_w,
            marker_indices=self.stage,
        )


@configclass
class CurriculumGoalCommandCfg(CommandTermCfg):
    """Configuration for :class:`CurriculumGoalCommand`."""

    class_type: type = CurriculumGoalCommand

    asset_name: str = "robot"
    """Name of the robot articulation in the scene."""

    object_name: str = "object"
    """Name of the rigid object the policy must move to the goal."""

    # --- Stage 0 (lift) ---
    lift_height: float = 0.10
    """Height (in robot root frame z) at which the lift goal sits, directly above the
    cube's spawn xy. Cube spawns at z ~= 0.015 so 0.10 = real lift of ~8.5cm."""

    # --- Stage 1 (carry) ---
    carry_height: float = 0.15
    """Robot-root-frame z height for stage 1.

    Stage 1 uses ``(place_goal.x, place_goal.y, carry_height)`` so the target is
    directly above the final placement point.
    """

    # --- Stage 2 (place) ---
    place_goal: tuple[float, float, float] = (0.30, 0.10, 0.02)
    """Fixed table pose in robot root frame for stage 2 (z ~= cube half-height)."""

    advance_threshold: float = 0.03
    """Visible goal sphere radius."""

    object_contact_radius: float = 0.015
    """Approximate object surface radius used for goal-sphere contact.

    The SO-101 pick-place cube is 3 cm wide, so 0.015 m makes logical contact
    match the cube face touching the visible sphere instead of requiring the
    cube center to enter the sphere.
    """

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = MISSING  # type: ignore
    """Marker config with three sphere variants (one per stage). Indexed by stage.
    Tasks must set this explicitly in the env_cfg so marker styles stay configurable
    per task."""

    lock_stage: int | None = None
    """If set, freeze the per-env stage at this value (no advancement). Used by
    teleop to show only the final place sphere — operators want one explicit
    end goal, not the lift / carry / place chain that drives RL training.
    ``None`` (the default) keeps the curriculum chain active for RL."""


__all__ = [
    "CurriculumGoalCommand",
    "CurriculumGoalCommandCfg",
]
