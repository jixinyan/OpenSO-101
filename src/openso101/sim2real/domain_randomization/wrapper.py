# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import gymnasium as gym
import torch

from .action import ActionDelayBuffer


class ActionDRWrapper(gym.Wrapper):
    def __init__(self, env, *, max_delay=3, deadband=0.01, scale_range=(0.95, 1.05)):
        super().__init__(env)
        self.core = ActionDelayBuffer(
            env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim,
            max_delay=max_delay, min_delay=int(max_delay > 0),
            device=env.unwrapped.device, seed=env.unwrapped.cfg.seed,
        )
        self.deadband = deadband
        self.scale_range = scale_range
        self.scale = torch.ones((self.core.num_envs, self.core.action_dim), device=self.core.device)
        self._reset_parameters(None)

    def _reset_parameters(self, env_ids):
        self.core.reset(env_ids)
        ids = torch.arange(self.core.num_envs, device=self.core.device) if env_ids is None else env_ids
        self.scale[ids] = self.scale_range[0] + torch.rand(
            (len(ids), self.core.action_dim), device=self.core.device,
        ) * (self.scale_range[1] - self.scale_range[0])

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._reset_parameters(None)
        return result

    def step(self, actions):
        adjusted = actions * self.scale
        adjusted = torch.where(adjusted.abs() < self.deadband, 0, adjusted)
        result = self.env.step(self.core.step(adjusted))
        ids = (result[2] | result[3]).nonzero(as_tuple=False).flatten()
        if ids.numel():
            self._reset_parameters(ids)
        return result
