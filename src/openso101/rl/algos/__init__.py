# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""RL algorithm modules.

PPO is a real thin wrapper over rsl_rl. SAC is a skeleton (sub-project E).
The `ALGOS` dict provides the algorithm-name → class lookup used by the CLI.
"""

from .ppo import PPO
from .sac import SAC

ALGOS = {
    "ppo": PPO,
    "sac": SAC,
}

__all__ = ["PPO", "SAC", "ALGOS"]
