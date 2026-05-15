# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""OpenSO-101: SOTA open-source robot learning framework for the SO-101."""

__version__ = "0.1.0"

# Trigger gym registration of built-in example tasks on package import.
try:
    from . import tasks  # noqa: F401
except ModuleNotFoundError:
    pass  # isaaclab not installed; submodules (teleop, robots) still importable
