# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Any


def _import_act_class() -> type:
    """通过 LeRobot factory 读取 ACTPolicy。"""
    from lerobot.policies.factory import get_policy_class

    return get_policy_class("act")


class _ACTProxy:
    """调用 ACTPolicy(cfg) 时构造并返回实际 LeRobot 模型。"""

    def __new__(cls, *args, **kwargs):
        real_cls = _import_act_class()
        return real_cls(*args, **kwargs)


ACTPolicy: Any = _ACTProxy


def load_act_policy(path: str | Path, *, device: str | None = None):
    """根据 checkpoint 配置读取模型及 processors。"""
    from .factory import load_policy

    return load_policy(path, device=device)


__all__ = ["ACTPolicy", "load_act_policy"]
