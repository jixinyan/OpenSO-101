# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Base env cfg for OpenSO-101 tasks.

`OpenSO101EnvCfg` extends Isaac Lab's `ManagerBasedRLEnvCfg` with three
variant-configuration hooks that the env factory (registered via
`openso101.envs.register_task`) calls after `__init__` based on
`gym.make()` kwargs:

* `configure_action_mode(mode)`  - "rl" (default) or "teleop".
* `configure_cameras(enabled)`   - attach wrist+overhead cameras.
* `configure_play(enabled)`      - shrink num_envs, disable obs noise.

Tasks override individual hooks only when they need behavior different
from the defaults.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.utils import configclass

from openso101.robots.so101 import SO101_SIM_JOINT_NAMES
from openso101.robots.so101.cameras import overhead_camera_cfg, wrist_camera_cfg


class UnsupportedVariantError(RuntimeError):
    """Raised by `configure_*` to signal a variant isn't implemented for a task."""


@configclass
class TeleopActionsCfg:
    """Absolute joint-position action manager for teleop semantics."""

    joint_positions: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(SO101_SIM_JOINT_NAMES),
        use_default_offset=False,
    )


_PLAY_NUM_ENVS_CAP = 16


@configclass
class OpenSO101EnvCfg(ManagerBasedRLEnvCfg):
    """Base cfg for OpenSO-101 tasks."""

    def configure_action_mode(self, mode: str) -> None:
        """`'rl'` is a no-op; `'teleop'` swaps in absolute joint targets."""
        if mode == "rl":
            return
        if mode == "teleop":
            self.actions = TeleopActionsCfg()
            return
        raise UnsupportedVariantError(
            f"action_mode={mode!r} not supported; expected 'rl' or 'teleop'."
        )

    def configure_cameras(self, enabled: bool) -> None:
        """Attach overhead + wrist cameras when `enabled`."""
        if not enabled:
            return
        self.scene.overhead_camera = overhead_camera_cfg()
        self.scene.wrist_camera = wrist_camera_cfg()

    def configure_play(self, enabled: bool) -> None:
        """Shrink num_envs and disable observation corruption for eval."""
        if not enabled:
            return
        self.scene.num_envs = min(self.scene.num_envs, _PLAY_NUM_ENVS_CAP)
        policy_obs = getattr(getattr(self, "observations", None), "policy", None)
        if policy_obs is not None and hasattr(policy_obs, "enable_corruption"):
            policy_obs.enable_corruption = False
