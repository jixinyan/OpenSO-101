# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Observation-space domain randomization (joint-state noise, etc).

SKELETON: see sub-project B for implementation.
"""

from __future__ import annotations


def attach_observation_dr(*args, **kwargs):
    """Attach observation-space DR to a task's EventCfg."""
    raise NotImplementedError(
        "Observation DR is part of sub-project B. See "
        "docs/superpowers/specs/2026-05-13-openso101-refactor-design.md § 13."
    )


__all__ = ["attach_observation_dr"]
