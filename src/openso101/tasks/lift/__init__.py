# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Lift example task — built-in OpenSO-101 task."""

from openso101.envs import register_task

from .lift_env_cfg import LiftEnvCfg

register_task(
    "OpenSO101-Lift-v0",
    agent_cfgs={
        "rsl_rl_cfg_entry_point": "openso101.tasks.lift.agents.rsl_rl_ppo_cfg:LiftPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": "openso101.tasks.lift.agents.rsl_rl_distillation_cfg:LiftDistillationRunnerCfg",
    },
)(LiftEnvCfg)
