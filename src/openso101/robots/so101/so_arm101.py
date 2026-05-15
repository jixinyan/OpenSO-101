# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""SO-101 ArticulationCfgs (RL and teleop) and USD spawn helpers.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/robots/trs_so101/so_arm101.py`
lands. The canonical robot asset is `assets/so101/usd/SO-ARM101-USD.usd`,
which will be copied into the repo when porting completes.

Note: the legacy code currently resolves the USD via a sibling
`isaac_so_arm101` checkout outside the repo. The real port should
either bundle the USD inside `openso101/assets/so101/usd/` or accept
the external path as documented configuration.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Contact-material and collision constants (TODO: fill from source on port)
# ---------------------------------------------------------------------------

SO101_GRIPPER_CONTACT_STATIC_FRICTION: float = 0.0  # TODO: port from source
SO101_GRIPPER_CONTACT_DYNAMIC_FRICTION: float = 0.0  # TODO: port from source
SO101_GRIPPER_CONTACT_OFFSET: float = 0.0  # TODO: port from source
SO101_GRIPPER_REST_OFFSET: float = 0.0  # TODO: port from source

# ---------------------------------------------------------------------------
# Derived init-state constants (TODO: fill from source on port)
# ---------------------------------------------------------------------------

SO101_USD_TABLETOP_ROOT_Z: float = 0.0  # TODO: computed at port time via _usd_bounds.tabletop_root_z
"""``init_state.pos.z`` that places the SO101 base bottom on a table top at world z=0."""

SO101_CANONICAL_INIT_JOINT_POS: dict[str, float] = {}  # TODO: port from source
"""Canonical SO101 reset posture shared by every task that uses ``SO_ARM101_CFG``."""

# ---------------------------------------------------------------------------
# ArticulationCfg placeholders
# ---------------------------------------------------------------------------

# Placeholder. Will hold the configured ArticulationCfg for RL training once ported.
SO_ARM101_CFG = None  # type: ignore[assignment]

# Placeholder. Will hold the teleop-tuned ArticulationCfg (softer actuators) once ported.
SO_ARM101_TELEOP_CFG = None  # type: ignore[assignment]


def so101_usd_path() -> Path:
    """Return the absolute path to the canonical SO-ARM101 USD asset.

    Source reference:
    /data/safe_sim2real/src/safe_sim2real/robots/trs_so101/so_arm101.py
    """
    raise NotImplementedError(
        "so101_usd_path not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/robots/trs_so101/so_arm101.py"
    )


def spawn_so101_usd_with_safe_collisions(
    prim_path: str,
    cfg,
    translation: "tuple[float, float, float] | None" = None,
    orientation: "tuple[float, float, float, float] | None" = None,
    **kwargs,
):
    """Spawn the upstream SO-ARM101 USD with pre-spawned colliders for gripper contacts.

    De-instances the gripper and jaw collision groups, cooks them as convex
    hulls, and assigns explicit contact offsets plus high-friction contact
    material at spawn time.

    Source reference:
    /data/safe_sim2real/src/safe_sim2real/robots/trs_so101/so_arm101.py
    """
    raise NotImplementedError(
        "spawn_so101_usd_with_safe_collisions not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/robots/trs_so101/so_arm101.py"
    )
