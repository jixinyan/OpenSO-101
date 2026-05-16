# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Diffusion policy — thin wrapper over LeRobot.

See `openso101.il.policies.act` for the rationale: we re-export LeRobot's
maintained Diffusion implementation rather than maintain a duplicate.

This module is **fully functional**: `DiffusionPolicy(cfg)` builds a real
trainable model, `load_diffusion_policy(path)` loads a checkpoint trained
by `lerobot.scripts.train --policy.type diffusion`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _import_diffusion_class() -> type:
    from lerobot.policies.factory import get_policy_class

    return get_policy_class("diffusion")


class _DiffusionProxy:
    """Lazy proxy — see `_ACTProxy` for the rationale."""

    def __new__(cls, *args, **kwargs):
        real_cls = _import_diffusion_class()
        return real_cls(*args, **kwargs)


DiffusionPolicy: Any = _DiffusionProxy


def load_diffusion_policy(path: str | Path, *, device: str | None = None):
    """Load a Diffusion checkpoint trained by LeRobot."""
    from .factory import load_policy

    return load_policy(path, device=device)


__all__ = ["DiffusionPolicy", "load_diffusion_policy"]
