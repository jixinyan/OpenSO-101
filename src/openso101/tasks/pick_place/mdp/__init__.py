# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Pick-and-place MDP -- 3-stage curriculum (lift -> carry -> place).

Reuses Isaac Lab's base + lift mdp namespaces and adds:
- :class:`CurriculumGoalCommand` / :class:`CurriculumGoalCommandCfg` for the
  per-env staged goal that drives the green sphere marker.
- staged goal-distance rewards plus sparse stage-completion bonuses.
- :func:`curriculum_complete` termination (only stage-2 success ends episodes).
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
    cube_to_curriculum_stage_goal,
    cube_to_curriculum_stage_goal_height_gated,
    cube_to_curriculum_stage_goal_when_grasped,
    object_is_lifted_when_grasped,
    stage_completion_bonus,
)
from .terminations import curriculum_complete  # noqa: F401
