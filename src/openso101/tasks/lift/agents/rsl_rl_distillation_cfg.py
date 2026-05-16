# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Distillation rsl_rl runner cfg for the Lift task."""

from __future__ import annotations

from openso101.tasks.shared.rsl_rl_distillation_cfg import (
    make_distillation_runner_cfg,
)


LiftDistillationRunnerCfg = make_distillation_runner_cfg(
    experiment_name="lift_distillation",
    max_iterations=1500,
)
