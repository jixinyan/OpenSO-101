# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from rsl_rl.algorithms import PPO as _RslRlPPO


paradigm = "on_policy"


class PPO(_RslRlPPO):
    pass


__all__ = ["PPO", "paradigm"]
