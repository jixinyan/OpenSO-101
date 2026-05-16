# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Distillation rsl_rl runner cfg for the Stack task."""

from __future__ import annotations

from openso101.tasks.shared.rsl_rl_distillation_cfg import (
    make_distillation_runner_cfg,
)


StackDistillationRunnerCfg = make_distillation_runner_cfg(
    experiment_name="stack_distillation",
    max_iterations=1500,
)
