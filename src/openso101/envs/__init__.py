# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""OpenSO-101 env base classes and task-registration helpers."""

from .base import OpenSO101EnvCfg, UnsupportedVariantError
from .registry import register_task

__all__ = ["OpenSO101EnvCfg", "UnsupportedVariantError", "register_task"]
