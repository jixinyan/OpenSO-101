# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import math
import os
import statistics

from rsl_rl.runners import OnPolicyRunner as _RslRlRunner


class OnPolicyRunner(_RslRlRunner):
    pass


class BestCheckpointRunner(_RslRlRunner):
    _SUCCESS_TIE_TOL = 1e-6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_mean_reward = float("-inf")
        self._best_success = float("-inf")
        self._best_success_reward = float("-inf")

    @staticmethod
    def _mean_success(locs):
        episodes = locs["ep_infos"]
        if not episodes:
            return None
        values = []
        for episode in episodes:
            value = episode["Episode_Termination/success"]
            if hasattr(value, "mean"):
                value = value.float().mean() if hasattr(value, "float") else value.mean()
            if hasattr(value, "item"):
                value = value.item()
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("训练日志的 success 数值必须位于 [0, 1]")
            values.append(value)
        return statistics.mean(values)

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        if not self.log_dir or not locs["rewbuffer"]:
            return
        success = self._mean_success(locs)
        if success is None:
            return
        reward = statistics.mean(locs["rewbuffer"])
        if not math.isfinite(reward):
            raise ValueError("训练日志的 reward 必须为有限数值")
        if reward > self._best_mean_reward:
            self._best_mean_reward = reward
            self.save(os.path.join(self.log_dir, "model_best_reward.pt"))
        accepted = (success > self._best_success + self._SUCCESS_TIE_TOL
                    or (abs(success - self._best_success) <= self._SUCCESS_TIE_TOL
                        and reward > self._best_success_reward))
        if accepted:
            self._best_success = max(success, self._best_success)
            self._best_success_reward = reward
            self.save(os.path.join(self.log_dir, "model_best.pt"))
            print(f"训练日志模型选择：iteration={locs['it']}，success={success:.2%}，reward={reward:+.2f}")


__all__ = ["OnPolicyRunner", "BestCheckpointRunner"]
