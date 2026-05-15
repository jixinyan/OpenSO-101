# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import pytest

pytest.importorskip("isaaclab")
import gymnasium as gym

from openso101.envs import OpenSO101EnvCfg, register_task


@register_task("OpenSO101Test-Dummy-v0")
class _DummyCfg(OpenSO101EnvCfg):
    """Minimal task used for register_task tests."""


def test_register_task_registers_gym_id():
    assert "OpenSO101Test-Dummy-v0" in gym.envs.registry


def test_register_task_passes_default_kwargs():
    spec = gym.envs.registry["OpenSO101Test-Dummy-v0"]
    assert spec.kwargs.get("action_mode", "rl") == "rl"
    assert spec.kwargs.get("cameras", False) is False
    assert spec.kwargs.get("play", False) is False


def test_factory_applies_configure_calls():
    called = []

    class _TrackingCfg(OpenSO101EnvCfg):
        def configure_action_mode(self, mode):
            called.append(("action", mode))

        def configure_cameras(self, enabled):
            called.append(("cameras", enabled))

        def configure_play(self, enabled):
            called.append(("play", enabled))

    register_task("OpenSO101Test-Tracking-v0")(_TrackingCfg)

    spec = gym.envs.registry["OpenSO101Test-Tracking-v0"]
    # The entry_point is the factory function we registered. Calling it directly
    # bypasses the ManagerBasedRLEnv construction in the factory's last line —
    # which is fine for this isolated unit test (we only verify the configure_* trace).
    # Use a mock for the env construction to avoid touching Isaac Lab's runtime.
    import unittest.mock as mock

    with mock.patch("openso101.envs.registry.ManagerBasedRLEnv") as mock_env:
        mock_env.return_value = "env_sentinel"
        result = spec.entry_point(action_mode="teleop", cameras=True, play=True)

    assert called == [("action", "teleop"), ("cameras", True), ("play", True)]
    assert result == "env_sentinel"


def test_register_task_rejects_non_openso101_cfg():
    class NotACfg:
        pass

    with pytest.raises(TypeError):
        register_task("OpenSO101Test-Bad-v0")(NotACfg)


def test_register_task_with_agent_cfgs_merges_kwargs():
    @register_task(
        "OpenSO101Test-WithAgents-v0",
        agent_cfgs={"rsl_rl_ppo_cfg_entry_point": "my_module:MyCfg"},
    )
    class _AgentsCfg(OpenSO101EnvCfg):
        pass

    spec = gym.envs.registry["OpenSO101Test-WithAgents-v0"]
    assert spec.kwargs.get("rsl_rl_ppo_cfg_entry_point") == "my_module:MyCfg"
