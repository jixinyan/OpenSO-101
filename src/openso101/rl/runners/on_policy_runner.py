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
        """Extends OnPolicyRunner with success-based checkpoint selection:

        - model_best.pt: saved whenever the running mean SUCCESS rate (the
          ``Episode_Termination/success`` per-term fraction logged by IsaacLab)
          hits a new high. Ties within 1e-6 are broken toward higher mean reward.
        - model_best_reward.pt: saved whenever the 100-episode mean reward hits a
          new high (the previous reward-best behavior, under a new filename).
        - If the success key is absent for this run (older task that does not log
          ``Episode_Termination/success``), model_best.pt falls back to the
          reward-based selection so nothing regresses.
        - Console banner printed at each new best so it's easy to spot in logs.
        """

        # Tie tolerance: success rates equal within this margin are treated as a
        # tie, broken toward the higher mean reward.
        _SUCCESS_TIE_TOL: float = 1e-6

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._best_mean_reward: float = float("-inf")
            self._best_success: float = float("-inf")
            # Mean reward at the iteration that set the current success-best, used
            # as the tie-break reference when success rates are (near) equal.
            self._best_success_reward: float = float("-inf")

        @staticmethod
        def _mean_success(locs: dict) -> float | None:
            """Mean ``Episode_Termination/success`` fraction over this iteration.

            Reads the per-episode term dicts the runner already collected in
            ``locs['ep_infos']`` (the standard rsl_rl key). Each present value is
            the fraction of episodes that ended via the ``success`` termination on
            that env-reset batch, so the mean across the list is the running mean
            success rate. Returns ``None`` when the key is absent / the list is
            empty (older tasks that do not log a success termination), signalling
            the caller to fall back to reward-based selection.
            """
            ep_infos = locs.get("ep_infos") or []
            key = "Episode_Termination/success"
            values: list[float] = []
            for ep_info in ep_infos:
                if not isinstance(ep_info, dict) or key not in ep_info:
                    continue
                val = ep_info[key]
                # Values may be python scalars or (0-d / 1-d) tensors.
                try:
                    if hasattr(val, "mean"):
                        val = val.float().mean() if hasattr(val, "float") else val.mean()
                    if hasattr(val, "item"):
                        val = val.item()
                    values.append(float(val))
                except (TypeError, ValueError):
                    continue
            if not values:
                return None
            return statistics.mean(values)

        def log(self, locs: dict, width: int = 80, pad: int = 35):
            super().log(locs, width, pad)

            if not self.log_dir or len(locs.get("rewbuffer", [])) == 0:
                return

            mean_reward = statistics.mean(locs["rewbuffer"])
            mean_success = self._mean_success(locs)

            # --- reward-best (preserved behavior, new filename) ---
            if mean_reward > self._best_mean_reward:
                self._best_mean_reward = mean_reward
                self.save(os.path.join(self.log_dir, "model_best_reward.pt"))
                banner = (
                    f" NEW BEST  iter {locs['it']:>5d}  reward {mean_reward:+.2f} "
                    f"-> model_best_reward.pt "
                )
                print(f"\n{'=' * width}")
                print(f"{banner:^{width}}")
                print(f"{'=' * width}\n")

            # --- primary selection: success rate (reward as fallback) ---
            if mean_success is None:
                # Older task with no success termination: keep model_best.pt on
                # the reward criterion so behavior does not regress.
                if mean_reward >= self._best_success_reward:
                    self._best_success_reward = mean_reward
                    self.save(os.path.join(self.log_dir, "model_best.pt"))
                return

            is_new_best = (mean_success > self._best_success + self._SUCCESS_TIE_TOL) or (
                abs(mean_success - self._best_success) <= self._SUCCESS_TIE_TOL
                and mean_reward > self._best_success_reward
            )
            if is_new_best:
                self._best_success = max(mean_success, self._best_success)
                self._best_success_reward = mean_reward
                self.save(os.path.join(self.log_dir, "model_best.pt"))
                banner = (
                    f" NEW BEST  iter {locs['it']:>5d}  success {mean_success:6.2%} "
                    f"(reward {mean_reward:+.2f}) -> model_best.pt "
                )
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
