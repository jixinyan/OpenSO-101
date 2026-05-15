# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Wrist + overhead camera factories for the SO-101 rig.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/robots/trs_so101/cameras.py`
lands.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Module-level pose / resolution constants (TODO: fill from source on port)
# ---------------------------------------------------------------------------

# Resolution
SO101_CAMERA_WIDTH: int = 0  # TODO: port from source
SO101_CAMERA_HEIGHT: int = 0  # TODO: port from source
SO101_CAMERA_DATA_TYPES: tuple[str, ...] = ()  # TODO: port from source

# Wrist camera
SO101_WRIST_CAMERA_PRIM_PATH: str = ""  # TODO: port from source
SO101_WRIST_CAMERA_POS: tuple[float, float, float] = (0.0, 0.0, 0.0)  # TODO: port from source
SO101_WRIST_CAMERA_ROT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # TODO: port from source
SO101_WRIST_CAMERA_FOCAL_LENGTH: float = 0.0  # TODO: port from source

# Overhead camera
SO101_OVERHEAD_CAMERA_PRIM_PATH: str = ""  # TODO: port from source
SO101_OVERHEAD_CAMERA_POS: tuple[float, float, float] = (0.0, 0.0, 0.0)  # TODO: port from source
SO101_OVERHEAD_CAMERA_ROT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # TODO: port from source
SO101_OVERHEAD_CAMERA_FOCAL_LENGTH: float = 0.0  # TODO: port from source


def wrist_camera_cfg(
    prim_path: str = SO101_WRIST_CAMERA_PRIM_PATH,
    width: int = SO101_CAMERA_WIDTH,
    height: int = SO101_CAMERA_HEIGHT,
    data_types: list[str] | None = None,
):
    """`TiledCameraCfg` parented to gripper_link, optical axis along the fingertip direction.

    Source reference:
    /data/safe_sim2real/src/safe_sim2real/robots/trs_so101/cameras.py
    """
    raise NotImplementedError(
        "wrist_camera_cfg not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/robots/trs_so101/cameras.py"
    )


def overhead_camera_cfg(
    prim_path: str = SO101_OVERHEAD_CAMERA_PRIM_PATH,
    width: int = SO101_CAMERA_WIDTH,
    height: int = SO101_CAMERA_HEIGHT,
    data_types: list[str] | None = None,
):
    """`TiledCameraCfg` pinned in the per-env frame, pitched down toward the workspace.

    Source reference:
    /data/safe_sim2real/src/safe_sim2real/robots/trs_so101/cameras.py
    """
    raise NotImplementedError(
        "overhead_camera_cfg not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/robots/trs_so101/cameras.py"
    )
