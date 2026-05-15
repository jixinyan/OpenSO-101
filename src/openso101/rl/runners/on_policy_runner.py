# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""On-policy runner — thin wrapper over rsl_rl's OnPolicyRunner."""

from __future__ import annotations

import os
import statistics

try:
    from rsl_rl.runners import OnPolicyRunner as _RslRlRunner
except ImportError:
    _RslRlRunner = None  # type: ignore[assignment]


if _RslRlRunner is not None:
    class OnPolicyRunner(_RslRlRunner):
        """Alias for `rsl_rl.runners.OnPolicyRunner`."""

    class BestCheckpointRunner(_RslRlRunner):
        """Extends OnPolicyRunner with:

        - model_best.pt: saved whenever the 100-episode mean reward hits a new high.
        - Console banner printed at each new best so it's easy to spot in logs.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._best_mean_reward: float = float("-inf")

        def log(self, locs: dict, width: int = 80, pad: int = 35):
            super().log(locs, width, pad)

            if self.log_dir and len(locs.get("rewbuffer", [])) > 0:
                mean_reward = statistics.mean(locs["rewbuffer"])
                if mean_reward > self._best_mean_reward:
                    self._best_mean_reward = mean_reward
                    best_path = os.path.join(self.log_dir, "model_best.pt")
                    self.save(best_path)
                    banner = f" NEW BEST  iter {locs['it']:>5d}  reward {mean_reward:+.2f} -> model_best.pt "
                    print(f"\n{'=' * width}")
                    print(f"{banner:^{width}}")
                    print(f"{'=' * width}\n")
else:
    class OnPolicyRunner:
        """rsl_rl not installed — OnPolicyRunner unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "rsl_rl is required for openso101.rl.runners.OnPolicyRunner."
            )

    class BestCheckpointRunner:
        """rsl_rl not installed — BestCheckpointRunner unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "rsl_rl is required for openso101.rl.runners.BestCheckpointRunner."
            )


__all__ = ["OnPolicyRunner", "BestCheckpointRunner"]
