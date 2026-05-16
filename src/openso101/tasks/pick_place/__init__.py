# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""PickPlace example task — built-in OpenSO-101 task."""

from openso101.envs import register_task

from .pick_place_env_cfg import PickPlaceEnvCfg

register_task(
    "OpenSO101-PickPlace-v0",
    agent_cfgs={
        "rsl_rl_cfg_entry_point": "openso101.tasks.pick_place.agents.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": "openso101.tasks.pick_place.agents.rsl_rl_distillation_cfg:PickPlaceDistillationRunnerCfg",
    },
)(PickPlaceEnvCfg)
