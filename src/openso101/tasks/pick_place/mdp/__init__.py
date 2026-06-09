# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Pick-and-lift MDP -- sentinel-style single-goal delta shaping.

Reuses Isaac Lab's base + lift mdp namespaces and adds:
- :class:`CurriculumGoalCommand` / :class:`CurriculumGoalCommandCfg`, frozen to
  a single goal via ``lock_stage`` (RL: air carry goal; teleop: table place
  goal), driving the green sphere marker.
- :func:`pregrasp_approach_shaping` / :func:`carry_to_goal_shaping`
  delta-distance rewards (progress, not position).
- :func:`reached_goal_while_grasped` termination: success requires the cube in
  the goal sphere AND a contact-confirmed grasp.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.manipulation.lift.mdp import *  # noqa: F401, F403

from openso101.tasks.shared.rewards import object_reached_goal_in_air  # noqa: F401

from .curriculum_goal_command import (  # noqa: F401
    CurriculumGoalCommand,
    CurriculumGoalCommandCfg,
)
from .grasp import grasped_reward, object_grasped_by_jaws  # noqa: F401
from .rewards import (  # noqa: F401
    carry_to_goal_shaping,
    pregrasp_approach_shaping,
)
from .terminations import reached_goal_while_grasped  # noqa: F401
