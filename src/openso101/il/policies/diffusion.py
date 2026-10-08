# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Any


def _import_diffusion_class() -> type:
    from lerobot.policies.factory import get_policy_class

    return get_policy_class("diffusion")


class _DiffusionProxy:
    """调用 DiffusionPolicy(cfg) 时构造并返回实际 LeRobot 模型。"""

    def __new__(cls, *args, **kwargs):
        real_cls = _import_diffusion_class()
        return real_cls(*args, **kwargs)


DiffusionPolicy: Any = _DiffusionProxy


def load_diffusion_policy(path: str | Path, *, device: str | None = None):
    """根据 checkpoint 配置读取模型及 processors。"""
    from .factory import load_policy

    return load_policy(path, device=device)


__all__ = ["DiffusionPolicy", "load_diffusion_policy"]
