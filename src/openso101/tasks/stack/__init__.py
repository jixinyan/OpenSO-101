# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Stack example task — built-in OpenSO-101 task."""

from openso101.envs import register_task

from .stack_env_cfg import StackEnvCfg

register_task(
    "OpenSO101-Stack-v0",
    agent_cfgs={
        "rsl_rl_cfg_entry_point": "openso101.tasks.stack.agents.rsl_rl_ppo_cfg:StackPPORunnerCfg",
    },
)(StackEnvCfg)
