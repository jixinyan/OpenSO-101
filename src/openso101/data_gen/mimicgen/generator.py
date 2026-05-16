# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""MimicGen integration for SO-101 demo augmentation.

DEFERRED. The CLI surface (`openso101 data generate --backend mimicgen ...`)
is wired so call sites are stable, but the generator body is intentionally
not implemented until the RL + IL pipelines are fully validated end-to-end
on human-collected teleop data. See `docs/guides/il.md` for the current
data-flow story (teleop HDF5 → LeRobot dataset → ACT/Diffusion training).
"""

from __future__ import annotations


def generate_with_mimicgen(*args, **kwargs):
    """Generate synthetic demos from a small seed set using MimicGen."""
    raise NotImplementedError(
        "MimicGen-based synthetic data generation is deferred. Collect demos "
        "via `openso101 il record` and train via `openso101 il train` instead."
    )


__all__ = ["generate_with_mimicgen"]
