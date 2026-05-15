# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Camera configurations for the SO-ARM101 sim2real rig.

The real hardware has two cameras:
  * An overhead camera fixed in the world between the leader and follower arms
    and in front of them, looking toward the shared workspace.
  * A wrist camera rigidly mounted on the follower arm's gripper.

These factory functions return fresh ``TiledCameraCfg`` objects per scene so
that the mutable spawn / offset configs are not shared across tasks.  The
numeric poses are centralized here so calibration changes happen in exactly
one place.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg

# Keep resolutions modest so GPU-parallel rendering stays cheap during vision
# runs. 128x128 is a common distillation target and fits ResNet-style encoders
# after a light upsample. Override these in task configs if you need more detail.
SO101_CAMERA_WIDTH = 128
SO101_CAMERA_HEIGHT = 128
SO101_CAMERA_DATA_TYPES = ("rgb",)

# Camera poses are expressed in the "world" camera convention, where the camera
# local +X axis is the optical axis and +Z is up.  Positions are metres and
# rotations are quaternions in (w, x, y, z).
#
# Overhead rig layout: the real setup has the follower base at (0, 0), the
# leader base at (2, 0), and a flexible overhead camera roughly at
# (1.0, 1.25, 0.8) aimed between the arms.  Because the sim models only the
# follower, we scale that geometry down so the follower and its workspace
# fill the frame — the camera is flexible in reality, so sim placement can
# be re-tuned per-task without breaking sim2real assumptions.
SO101_WRIST_CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Robot/gripper/gripper_cam"
# Upstream SO101 USD camera mount. The USD contains the physical camera mount
# geometry on the gripper; this sensor is spawned at that mount point.
SO101_WRIST_CAMERA_POS = (-0.005, 0.06, -0.062)
SO101_WRIST_CAMERA_ROT = (0.9238795, -0.3826834, 0.0, 0.0)
SO101_WRIST_CAMERA_FOCAL_LENGTH = 2.4

SO101_OVERHEAD_CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/overhead_cam"
# GUI-calibrated: ~0.30 m above and just in front of the follower base,
# pitched 10° down and yawed 90° so the optical axis looks along +Y across
# the workspace.  Equivalent to Isaac Sim ``rotateXYZ = (0, 10, 90)``.
SO101_OVERHEAD_CAMERA_POS = (0.39028, -0.01593, 0.30459)
SO101_OVERHEAD_CAMERA_ROT = (0.71660, 0.06054, 0.06269, 0.69201)
SO101_OVERHEAD_CAMERA_FOCAL_LENGTH = 10.0


def _default_pinhole(focal_length: float) -> sim_utils.PinholeCameraCfg:
    # Pinhole intrinsics matching a wide-FOV USB webcam.  HFOV depends on
    # focal length: focal=8 -> HFOV ~105°, focal=10 -> HFOV ~93°.
    return sim_utils.PinholeCameraCfg(
        focal_length=focal_length,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 2.0),
    )


def _upstream_wrist_camera() -> sim_utils.FisheyeCameraCfg:
    return sim_utils.FisheyeCameraCfg(
        projection_type="fisheyeKannalaBrandtK3",
        fisheye_nominal_height=480,
        fisheye_nominal_width=640,
        fisheye_optical_centre_x=320,
        fisheye_optical_centre_y=240,
        fisheye_max_fov=170,
        fisheye_polynomial_a=0,
        fisheye_polynomial_b=0.0028,
        fisheye_polynomial_c=0,
        fisheye_polynomial_d=0,
        fisheye_polynomial_e=0,
        fisheye_polynomial_f=0,
        f_stop=180,
        focal_length=SO101_WRIST_CAMERA_FOCAL_LENGTH,
        focus_distance=0.05,
        clipping_range=(0.01, 2.0),
    )


def wrist_camera_cfg(
    prim_path: str = SO101_WRIST_CAMERA_PRIM_PATH,
    width: int = SO101_CAMERA_WIDTH,
    height: int = SO101_CAMERA_HEIGHT,
    data_types: list[str] | None = None,
) -> TiledCameraCfg:
    """Wrist camera rigidly attached to the upstream USD gripper camera mount.

    The prim path is parented under ``Robot/gripper`` so the sensor follows
    the camera mount geometry included in the upstream SO101 USD model.
    """

    return TiledCameraCfg(
        prim_path=prim_path,
        offset=TiledCameraCfg.OffsetCfg(
            pos=SO101_WRIST_CAMERA_POS,
            rot=SO101_WRIST_CAMERA_ROT,
            convention="opengl",
        ),
        data_types=list(data_types) if data_types is not None else list(SO101_CAMERA_DATA_TYPES),
        spawn=_upstream_wrist_camera(),
        width=width,
        height=height,
    )


def overhead_camera_cfg(
    prim_path: str = SO101_OVERHEAD_CAMERA_PRIM_PATH,
    width: int = SO101_CAMERA_WIDTH,
    height: int = SO101_CAMERA_HEIGHT,
    data_types: list[str] | None = None,
) -> TiledCameraCfg:
    """Fixed overhead camera sitting between the leader and follower arms.

    Pose is expressed in the per-environment local frame (``{ENV_REGEX_NS}``)
    so every parallel env gets its own camera above its own workspace.  The
    default pose (``x=0.3, z=0.5``) places it ~30 cm in front of the follower
    base and 50 cm up, pitched ~63° downward so the table at ``x≈0.5`` fills
    the frame.
    """

    return TiledCameraCfg(
        prim_path=prim_path,
        # Rotate +63.43° about +Y so +X_cam points from (0.3, 0, 0.5) toward
        # the table centre (~0.5, 0, 0.1).  Quaternion (w, x, y, z) =
        # (cos(θ/2), 0, sin(θ/2), 0) with θ/2 ≈ 31.72°.
        offset=TiledCameraCfg.OffsetCfg(
            pos=SO101_OVERHEAD_CAMERA_POS,
            rot=SO101_OVERHEAD_CAMERA_ROT,
            convention="opengl",
        ),
        data_types=list(data_types) if data_types is not None else list(SO101_CAMERA_DATA_TYPES),
        spawn=_default_pinhole(SO101_OVERHEAD_CAMERA_FOCAL_LENGTH),
        width=width,
        height=height,
    )
