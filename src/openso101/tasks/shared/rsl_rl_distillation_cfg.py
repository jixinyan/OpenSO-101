# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Shared factory for per-task rsl_rl Distillation runner configs.

Distillation in `rsl_rl` is teacher → student knowledge transfer: an
already-trained teacher policy (typically a PPO checkpoint produced by
`openso101 rl train --algo ppo --task <task>`) is loaded read-only and a
fresh student policy is trained to mimic the teacher's action
distribution. Both networks must have a compatible architecture; the
student is what you eventually deploy.

We expose a single `make_distillation_runner_cfg(experiment_name, ...)`
factory so per-task config files stay minimal (just an experiment name +
optional overrides). Each task registers the result under the
`rsl_rl_distillation_cfg_entry_point` key so `openso101 rl train --algo
distillation --task <task>` picks it up automatically.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)

from openso101.tasks.shared.rl_defaults import (
    SO101_DISTILL_ACTIVATION,
    SO101_DISTILL_GRADIENT_LENGTH,
    SO101_DISTILL_HIDDEN_DIMS,
    SO101_DISTILL_INIT_NOISE_STD,
    SO101_DISTILL_LEARNING_RATE,
    SO101_DISTILL_LOSS_TYPE,
    SO101_DISTILL_MAX_GRAD_NORM,
    SO101_DISTILL_NOISE_STD_TYPE,
    SO101_DISTILL_NUM_LEARNING_EPOCHS,
    SO101_DISTILL_NUM_STEPS_PER_ENV,
)


def make_distillation_runner_cfg(
    *,
    experiment_name: str,
    max_iterations: int = 2000,
    save_interval: int = 100,
    student_hidden_dims: list[int] | None = None,
    teacher_hidden_dims: list[int] | None = None,
) -> RslRlDistillationRunnerCfg:
    """Build a Distillation runner cfg with SO-101 defaults.

    Parameters
    ----------
    experiment_name:
        Name used by the rsl_rl logger for the run directory.
    max_iterations:
        Number of training iterations.
    save_interval:
        Checkpoint cadence.
    student_hidden_dims / teacher_hidden_dims:
        Optional architecture overrides. Both default to
        `SO101_DISTILL_HIDDEN_DIMS` so a freshly-trained PPO teacher loads
        without architecture mismatch.
    """
    student_dims = list(student_hidden_dims or SO101_DISTILL_HIDDEN_DIMS)
    teacher_dims = list(teacher_hidden_dims or SO101_DISTILL_HIDDEN_DIMS)

    return RslRlDistillationRunnerCfg(
        num_steps_per_env=SO101_DISTILL_NUM_STEPS_PER_ENV,
        max_iterations=max_iterations,
        save_interval=save_interval,
        experiment_name=experiment_name,
        empirical_normalization=True,
        policy=RslRlDistillationStudentTeacherCfg(
            init_noise_std=SO101_DISTILL_INIT_NOISE_STD,
            noise_std_type=SO101_DISTILL_NOISE_STD_TYPE,
            student_obs_normalization=True,
            teacher_obs_normalization=True,
            student_hidden_dims=student_dims,
            teacher_hidden_dims=teacher_dims,
            activation=SO101_DISTILL_ACTIVATION,
        ),
        algorithm=RslRlDistillationAlgorithmCfg(
            num_learning_epochs=SO101_DISTILL_NUM_LEARNING_EPOCHS,
            learning_rate=SO101_DISTILL_LEARNING_RATE,
            gradient_length=SO101_DISTILL_GRADIENT_LENGTH,
            max_grad_norm=SO101_DISTILL_MAX_GRAD_NORM,
            loss_type=SO101_DISTILL_LOSS_TYPE,
        ),
    )


__all__ = ["make_distillation_runner_cfg"]
