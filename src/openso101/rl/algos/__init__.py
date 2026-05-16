# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""RL algorithm modules.

OpenSO-101 ships PPO via a thin wrapper around `rsl_rl.algorithms.PPO`. The
`ALGOS` dict provides the algorithm-name → class lookup used by the CLI.
"""

from .ppo import PPO

ALGOS = {
    "ppo": PPO,
}

__all__ = ["PPO", "ALGOS"]
