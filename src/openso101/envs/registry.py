# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Public extension API: `register_task` decorator for OpenSO-101 envs."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Type

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv

from .base import OpenSO101EnvCfg


def register_task(
    gym_id: str,
    /,
    *,
    agent_cfgs: Mapping[str, str] | None = None,
    default_action_mode: str = "rl",
    default_cameras: bool = False,
    default_play: bool = False,
) -> Callable[[Type[OpenSO101EnvCfg]], Type[OpenSO101EnvCfg]]:
    """Register `cfg_cls` with the gym registry under `gym_id`.

    The factory inserted into the gym spec accepts three optional kwargs
    (`action_mode`, `cameras`, `play`) and routes them to the corresponding
    `OpenSO101EnvCfg.configure_*` methods before constructing
    `ManagerBasedRLEnv`.

    Parameters
    ----------
    gym_id : str
        Positional-only gym ID, e.g. ``"OpenSO101-Lift-v0"``.
    agent_cfgs : optional mapping
        Extra kwargs merged into the gym spec (e.g. RL agent cfg entry points).
    default_action_mode, default_cameras, default_play
        Variant defaults baked into the gym spec; overridable per `gym.make()` call.

    Returns
    -------
    decorator
        Class decorator preserving the original class.
    """

    def _decorator(cfg_cls: Type[OpenSO101EnvCfg]) -> Type[OpenSO101EnvCfg]:
        if not (isinstance(cfg_cls, type) and issubclass(cfg_cls, OpenSO101EnvCfg)):
            raise TypeError(
                f"{cfg_cls!r} must be a subclass of OpenSO101EnvCfg "
                f"to be registered as an OpenSO-101 task."
            )

        def _factory(
            action_mode: str = default_action_mode,
            cameras: bool = default_cameras,
            play: bool = default_play,
            **kwargs: Any,
        ) -> ManagerBasedRLEnv:
            cfg = cfg_cls()
            cfg.configure_action_mode(action_mode)
            cfg.configure_cameras(cameras)
            cfg.configure_play(play)
            return ManagerBasedRLEnv(cfg=cfg, **kwargs)

        # Expose the cfg class via an `env_cfg_entry_point` string so Isaac
        # Lab's `parse_env_cfg` / `load_cfg_from_registry` can import-and-load
        # the cfg by name (the train/play scripts use that path).
        env_cfg_entry_point = f"{cfg_cls.__module__}:{cfg_cls.__name__}"
        spec_kwargs: dict[str, Any] = {
            "env_cfg_entry_point": env_cfg_entry_point,
            "action_mode": default_action_mode,
            "cameras": default_cameras,
            "play": default_play,
        }
        if agent_cfgs:
            spec_kwargs.update(agent_cfgs)

        gym.register(
            id=gym_id,
            entry_point=_factory,
            kwargs=spec_kwargs,
            disable_env_checker=True,
        )
        return cfg_cls

    return _decorator
