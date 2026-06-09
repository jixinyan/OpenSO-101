# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Pure-tensor reward-shaping primitives (no Isaac Lab dependency).

These helpers implement the *delta-distance* shaping pattern from the
sentinel ``PickAndLiftReward``: instead of rewarding closeness directly
(which lets a policy hover near the target and farm reward), we reward the
*reduction* in distance from one step to the next. Hovering yields zero;
moving toward the target yields positive; moving away yields negative.

The state machine that decides *when* a delta is valid (correct grasp mode,
not a fresh episode reset) is also factored out here so it can be unit
tested without launching a SimulationApp. The Isaac-Lab-coupled reward terms
in ``tasks/*/mdp/rewards.py`` gather the tensors and call into these.

This module lives at the top level (``openso101.shaping``) rather than under
``openso101.tasks`` precisely so it can be imported without Isaac Lab: the
``tasks`` package eagerly registers gym envs on import, which pulls in
``isaaclab``. Keeping these pure tensor primitives outside that package lets
the unit tests run in plain pytest on any machine.
"""

from __future__ import annotations

import torch


def shaping_valid_mask(
    mode_active: torch.Tensor,
    prev_mode_active: torch.Tensor,
    fresh_reset: torch.Tensor,
) -> torch.Tensor:
    """Per-env mask for when a delta-progress reward should apply.

    A delta is only meaningful when the env was in the *same* shaping mode on
    both the previous and current step (so the two distances are comparable),
    and the previous distance is not stale from a just-reset episode.

    Mirrors the sentinel reward's ``if self._was_grasping and prev is not
    None`` guards, vectorised across envs.

    Args:
        mode_active: bool ``[N]`` — this term's mode is active this step
            (e.g. ``grasping`` for carry shaping, ``~grasping`` for pregrasp).
        prev_mode_active: bool ``[N]`` — the mode was active last step.
        fresh_reset: bool ``[N]`` — the env's episode just (re)started, so the
            stored previous distance is stale and must not be differenced.

    Returns:
        bool ``[N]`` — True where the delta is valid.
    """
    return mode_active & prev_mode_active & (~fresh_reset)


def delta_progress_reward(
    prev_dist: torch.Tensor,
    cur_dist: torch.Tensor,
    coeff: float,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """``(prev_dist - cur_dist) * coeff`` where ``valid_mask`` else ``0``.

    Positive when the env got closer to the target, negative when it moved
    away, zero when it held position or the mask is False.

    Args:
        prev_dist: float ``[N]`` — distance to target on the previous step.
        cur_dist: float ``[N]`` — distance to target this step.
        coeff: scalar gain on the per-step distance reduction.
        valid_mask: bool ``[N]`` — from :func:`shaping_valid_mask`.

    Returns:
        float ``[N]`` — the per-env shaping reward.
    """
    delta = (prev_dist - cur_dist) * coeff
    return torch.where(valid_mask, delta, torch.zeros_like(delta))


def resolve_prev(prev: torch.Tensor | None, cur: torch.Tensor) -> torch.Tensor:
    """Return a usable ``prev`` tensor, initialising to ``cur`` when absent.

    On the first call (no stored state) or after a shape/device change, the
    previous distance is unknown; we seed it to the current distance so the
    first computed delta is zero (which the caller also masks out via
    ``fresh_reset`` anyway). This keeps the shaping numerically safe without
    special-casing ``None`` at every call site.
    """
    if prev is None or prev.shape != cur.shape or prev.device != cur.device:
        return cur.clone()
    return prev


__all__ = [
    "shaping_valid_mask",
    "delta_progress_reward",
    "resolve_prev",
]
