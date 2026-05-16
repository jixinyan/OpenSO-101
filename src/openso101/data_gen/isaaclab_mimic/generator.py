# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""IsaacLab-Mimic integration for SO-101 demo augmentation.

DEFERRED. The CLI surface (`openso101 data generate --backend isaaclab_mimic
...`) is wired so call sites are stable, but the generator body is
intentionally not implemented until the RL + IL pipelines are fully
validated end-to-end on human-collected teleop data.
"""

from __future__ import annotations


def generate_with_isaaclab_mimic(*args, **kwargs):
    """Generate synthetic demos from a small seed set using IsaacLab-Mimic."""
    raise NotImplementedError(
        "IsaacLab-Mimic-based synthetic data generation is deferred. Collect "
        "demos via `openso101 il record` and train via `openso101 il train`."
    )


__all__ = ["generate_with_isaaclab_mimic"]
