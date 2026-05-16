# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Observation-space domain randomization.

Adds per-step uniform noise to proprioceptive observation terms
(joint positions + velocities) so a policy trained in sim is robust
to encoder noise and quantization on real Feetech servos. Lives
alongside the physics + visual DR modules under the same
``sim2real.domain_randomization`` umbrella.

Wiring example::

    from openso101.sim2real.domain_randomization.observation import (
        attach_observation_dr,
    )

    cfg = PickPlaceEnvCfg()
    attach_observation_dr(cfg.observations.policy)
"""

from __future__ import annotations

from typing import Any


def attach_observation_dr(
    obs_group,
    *,
    joint_pos_noise: float = 0.02,
    joint_vel_noise: float = 0.05,
) -> None:
    """Attach uniform noise to standard proprio observation terms.

    The defaults are calibrated against the SO-101 Feetech encoder
    spec: ±0.02 rad (~±1°) covers typical position read noise, and
    ±0.05 rad/s covers single-tick velocity quantization. Override via
    the keyword args if your real hardware has tighter or looser noise.

    The noise applies to ``joint_pos`` and ``joint_vel`` observation
    terms on the supplied observation group (typically
    ``cfg.observations.policy``). Other terms (object pose, ee frame,
    images) are left alone — they're handled by separate DR modules.

    Silently no-ops on missing terms so this is safe to call across
    tasks with heterogeneous observation specs.
    """
    try:
        from isaaclab.utils.noise import UniformNoiseCfg
    except ImportError as exc:
        raise RuntimeError(
            "Isaac Lab must be installed for observation DR. Run "
            "`bash scripts/install.sh` from the repo root."
        ) from exc

    _set_noise_if_present(
        obs_group,
        "joint_pos",
        UniformNoiseCfg(n_min=-float(joint_pos_noise), n_max=float(joint_pos_noise)),
    )
    _set_noise_if_present(
        obs_group,
        "joint_vel",
        UniformNoiseCfg(n_min=-float(joint_vel_noise), n_max=float(joint_vel_noise)),
    )


def _set_noise_if_present(obs_group: Any, term_name: str, noise_cfg) -> None:
    term = getattr(obs_group, term_name, None)
    if term is None:
        return
    # ObsTerm exposes a `noise` field that the observation manager
    # applies after func() is called. Setting it is enough — Isaac Lab
    # handles the per-step sampling.
    setattr(term, "noise", noise_cfg)


__all__ = ["attach_observation_dr"]
