# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""LeRobot dataset adapter for OpenSO-101 IL training.

SKELETON: see sub-project C for implementation.
"""

from __future__ import annotations


def load_lerobot_dataset(*args, **kwargs):
    """Load a LeRobot dataset for IL training."""
    raise NotImplementedError(
        "load_lerobot_dataset is part of sub-project C (IL training). See "
        "docs/superpowers/specs/2026-05-13-openso101-refactor-design.md § 13."
    )


__all__ = ["load_lerobot_dataset"]
