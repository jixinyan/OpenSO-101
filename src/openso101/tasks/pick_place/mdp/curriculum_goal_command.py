# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""PickPlace task-local MDP curriculum goal command.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/curriculum_goal_command.py` lands.

Exports two symbols:
- ``CurriculumGoalCommand`` — a ``CommandTerm`` subclass that tracks per-env
  curriculum stage (Lift → Carry → Place) and exposes the active goal in robot
  root frame as the command tensor.
- ``CurriculumGoalCommandCfg`` — the matching ``@configclass`` / ``CommandTermCfg``
  subclass.  Requires Isaac Lab at import time; wrapped in a try/except so the
  skeleton stays importable without Isaac Lab installed (the name is still exported
  as ``None`` with a TODO note).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CurriculumGoalCommand:
    """Per-env stage tracker that exposes ``goal - in robot root frame`` as the command.

    Subclasses :class:`isaaclab.managers.CommandTerm` so it integrates
    seamlessly with the rest of Isaac Lab (reward functions and observations
    read it via ``env.command_manager.get_command(name)``; debug-vis spawns
    the marker automatically).

    SKELETON — instantiation raises NotImplementedError until the real port
    lands.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CurriculumGoalCommand not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/curriculum_goal_command.py"
        )

    def goal_for_stage(self, stage: int) -> "torch.Tensor":  # noqa: F821
        """Return the per-env goal position for a specific curriculum stage."""
        raise NotImplementedError(
            "CurriculumGoalCommand.goal_for_stage not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/curriculum_goal_command.py"
        )

    def is_touching_goal(
        self,
        cube_pos_b: "torch.Tensor",  # noqa: F821
        goal_pos_b: "torch.Tensor | None" = None,  # noqa: F821
        threshold: "float | None" = None,
    ) -> "torch.Tensor":  # noqa: F821
        """Return true when the cube surface touches the goal sphere."""
        raise NotImplementedError(
            "CurriculumGoalCommand.is_touching_goal not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/curriculum_goal_command.py"
        )

    @property
    def command(self) -> "torch.Tensor":  # noqa: F821
        """Goal position in robot root frame, shape (num_envs, 3)."""
        raise NotImplementedError(
            "CurriculumGoalCommand.command not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/curriculum_goal_command.py"
        )


# CurriculumGoalCommandCfg requires isaaclab.managers.CommandTermCfg and
# isaaclab.utils.configclass at class-definition time. Wrap the import so the
# skeleton stays importable without Isaac Lab.
# TODO: replace with the real @configclass subclass during the full port.
try:
    from dataclasses import MISSING

    from isaaclab.managers import CommandTermCfg
    from isaaclab.markers import VisualizationMarkersCfg
    from isaaclab.utils import configclass

    @configclass
    class CurriculumGoalCommandCfg(CommandTermCfg):
        """Configuration for :class:`CurriculumGoalCommand`.

        SKELETON — fields declare the public surface; the class_type and
        defaults will be filled in during the real port.
        """

        class_type: type = CurriculumGoalCommand

        asset_name: str = "robot"
        """Name of the robot articulation in the scene."""

        object_name: str = "object"
        """Name of the rigid object the policy must move to the goal."""

        lift_height: float = 0.10
        """Height (in robot root frame z) at which the lift goal sits."""

        carry_height: float = 0.15
        """Robot-root-frame z height for stage 1 (carry)."""

        place_goal: tuple = (0.20, 0.18, 0.02)
        """Fixed table pose in robot root frame for stage 2 (place)."""

        advance_threshold: float = 0.03
        """Visible goal sphere radius."""

        object_contact_radius: float = 0.015
        """Approximate object surface radius used for goal-sphere contact."""

        goal_pose_visualizer_cfg: VisualizationMarkersCfg = MISSING  # type: ignore
        """Marker config with three sphere variants (one per stage)."""

except ModuleNotFoundError:
    # Isaac Lab not installed — export a sentinel so callers can detect absence.
    # TODO: replace with the real @configclass during the full port.
    CurriculumGoalCommandCfg = None  # type: ignore


__all__ = [
    "CurriculumGoalCommand",
    "CurriculumGoalCommandCfg",
]
