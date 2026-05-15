# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Domain randomization layer.

SKELETON: physics module mirrors the legacy DR surface; action/obs/visual
modules are placeholders for sub-project B.
"""

from .action import attach_action_dr
from .observation import attach_observation_dr
from .physics import attach_all_physics_dr
from .visual import attach_visual_dr

__all__ = [
    "attach_action_dr",
    "attach_observation_dr",
    "attach_all_physics_dr",
    "attach_visual_dr",
]
