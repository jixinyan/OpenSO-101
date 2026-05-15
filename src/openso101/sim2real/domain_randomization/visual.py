# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Visual domain randomization (camera pose jitter, lighting, textures).

SKELETON: see sub-project B for implementation.
"""

from __future__ import annotations


def attach_visual_dr(*args, **kwargs):
    """Attach visual DR to a task's EventCfg."""
    raise NotImplementedError(
        "Visual DR is part of sub-project B. See "
        "docs/superpowers/specs/2026-05-13-openso101-refactor-design.md § 13."
    )


__all__ = ["attach_visual_dr"]
