# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Distillation rsl_rl runner cfg for the PickPlace task."""

from __future__ import annotations

from isaaclab.utils import configclass

from openso101.tasks.shared.rsl_rl_distillation_cfg import (
    make_distillation_runner_cfg,
)


# Per-task overrides go here. `configclass` lets us subclass and tweak any
# field if a task needs a non-default student/teacher architecture or a
# different iteration budget.
PickPlaceDistillationRunnerCfg = make_distillation_runner_cfg(
    experiment_name="pick_place_distillation",
    max_iterations=1500,
)
