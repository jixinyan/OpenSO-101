# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import pytest

pytest.importorskip("isaaclab")

from openso101.envs.base import OpenSO101EnvCfg, UnsupportedVariantError
from openso101.robots.so101 import SO101_SIM_JOINT_NAMES


class _MinimalCfg(OpenSO101EnvCfg):
    """Bare-bones cfg used only for unit-testing the base class."""


def _make_cfg():
    cfg = _MinimalCfg()
    return cfg


def test_configure_action_mode_rl_leaves_actions_unchanged():
    cfg = _make_cfg()
    original_actions = cfg.actions
    cfg.configure_action_mode("rl")
    assert cfg.actions is original_actions


def test_configure_action_mode_teleop_uses_absolute_joint_targets():
    cfg = _make_cfg()
    cfg.configure_action_mode("teleop")
    joint_positions = cfg.actions.joint_positions
    assert tuple(joint_positions.joint_names) == SO101_SIM_JOINT_NAMES
    assert joint_positions.use_default_offset is False


def test_configure_action_mode_invalid_raises():
    cfg = _make_cfg()
    with pytest.raises(UnsupportedVariantError):
        cfg.configure_action_mode("invalid")
