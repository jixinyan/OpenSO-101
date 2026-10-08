# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    if os.environ.get("OPENSO101_SKIP_ISAAC", "0") == "1":
        return
    from openso101.rl.gpu_scope import configure_visible_gpu

    configure_visible_gpu()
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    config._openso101_app_launcher = launcher  # type: ignore[attr-defined]


def pytest_unconfigure(config: pytest.Config) -> None:
    launcher = getattr(config, "_openso101_app_launcher", None)
    if launcher is not None:
        launcher.app.close()
