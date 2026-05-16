# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""IL policy re-exports.

OpenSO-101 does not maintain its own ACT / Diffusion implementations:
LeRobot ships maintained, SO-101-friendly versions of both. These modules
re-export them under a stable `openso101.il.policies.*` namespace so call
sites stay decoupled from LeRobot's internal layout and so future
SO-101-specific overrides can be layered here without churning consumers.
"""

from .act import ACTPolicy, load_act_policy
from .diffusion import DiffusionPolicy, load_diffusion_policy
from .factory import load_policy, policy_class

__all__ = [
    "ACTPolicy",
    "DiffusionPolicy",
    "load_act_policy",
    "load_diffusion_policy",
    "load_policy",
    "policy_class",
]
