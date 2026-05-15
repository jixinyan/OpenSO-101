# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""PPO algorithm for OpenSO-101.

Thin wrapper around `rsl_rl.algorithms.PPO`. The wrapper exists so future
implementations (custom PPO, alternative RL library) can swap in without
changing the CLI surface or task agent cfgs.
"""

from __future__ import annotations

try:
    from rsl_rl.algorithms import PPO as _RslRlPPO
except ImportError:
    _RslRlPPO = None  # type: ignore[assignment]


paradigm = "on_policy"


if _RslRlPPO is not None:
    class PPO(_RslRlPPO):
        """Alias for `rsl_rl.algorithms.PPO`. See base class for full API."""
else:
    class PPO:
        """rsl_rl not installed — PPO unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "rsl_rl is required for openso101.rl.algos.PPO. "
                "Install it via the project's standard install procedure."
            )


__all__ = ["PPO", "paradigm"]
