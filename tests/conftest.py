# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Session-scoped pytest fixture that launches Isaac Sim once.

Isaac Lab's `isaaclab.envs` submodule (and many sibling submodules) is
gated behind the Omniverse kit ABI, which only bootstraps after a
`SimulationApp` instance exists. Tests that import `openso101.envs`,
`openso101.tasks.*`, or any `isaaclab.envs.*` symbol will fail at
collection time without an active SimulationApp.

This conftest launches a headless `AppLauncher` once per session if
`isaaclab` is importable; otherwise it lets the tests' own
`pytest.importorskip("isaaclab")` skip them.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        # No isaaclab installed — leave alone; per-test importorskip will skip.
        return

    launcher = AppLauncher(headless=True)
    # Stash the handle on the config so we can close it later.
    config._openso101_app_launcher = launcher  # type: ignore[attr-defined]


def pytest_unconfigure(config: pytest.Config) -> None:
    launcher = getattr(config, "_openso101_app_launcher", None)
    if launcher is not None:
        launcher.app.close()
