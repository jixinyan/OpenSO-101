# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Deploy a trained LeRobot policy on the real SO-101 follower arm.

Architecture mirrors :func:`openso101.cli.il._cmd_play` so the same
checkpoint runs in sim OR on hardware without modification. Only the
observation source and action sink change:

  ============  ===========================  ============================
  Component     Sim (`il play`)              Real (`sim2real deploy`)
  ============  ===========================  ============================
  Joint obs     scene["robot"].data.joint_*  follower.get_observation()
  Camera obs    TiledCameraCfg buffers       OpenCV-backed USB cameras
  Action sink   env.step(actions)            follower.send_action(dict)
  ============  ===========================  ============================

The dataset's ``observation.state`` and ``action`` schemas are in
**LeRobot motor units** (``[-100, 100]``) — same on both paths, so the
trained policy needs zero re-calibration to transfer.

This module is import-light by design: heavy LeRobot + cv2 imports
happen inside the entrypoint so `openso101 sim2real --help` doesn't
require the dependencies. Mirrors the lazy-import pattern used
throughout `openso101.cli.*`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


# Camera names must match the dataset schema produced by
# ``OpenSO101HDF5TeleopRecorder`` so a policy trained on teleop data
# can be deployed on hardware without renaming inputs.
_REAL_CAMERA_NAMES: tuple[str, ...] = ("wrist_camera", "overhead_camera")


def deploy(args: argparse.Namespace) -> int:
    """Real-hardware deploy entrypoint dispatched from the CLI.

    Connects the follower arm + USB cameras, loads the LeRobot
    checkpoint, and runs an inference loop until ``--max-steps`` is
    reached or the user Ctrl+C's.
    """
    follower, cameras, policy = None, None, None
    try:
        follower = _connect_so101_follower(
            port=args.follower_port,
            robot_id=args.follower_id,
        )
        print(
            f"[INFO]: Connected SO101 follower '{args.follower_id}' "
            f"on {args.follower_port}."
        )

        cameras = _open_cameras(
            wrist_index=args.wrist_camera_index,
            overhead_index=args.overhead_camera_index,
            width=args.camera_width,
            height=args.camera_height,
        )
        print(f"[INFO]: Opened cameras: {list(cameras)}.")

        policy = _load_lerobot_policy(args.policy_path, device=args.device)
        print(f"[INFO]: Loaded policy from {args.policy_path}.")

        target_dt = 1.0 / float(max(1, args.fps))
        step = 0
        max_steps = int(args.max_steps) if args.max_steps is not None else None

        while True:
            loop_start = time.perf_counter()

            obs = _build_real_observation(follower, cameras)
            action = policy.select_action(obs)
            # Policy returns shape (1, 6) motor units; squeeze to a
            # plain list of 6 floats keyed by LeRobot joint names.
            action_np = action.detach().cpu().numpy().reshape(-1)
            action_dict = _action_array_to_lerobot_dict(action_np)

            follower.send_action(action_dict)

            step += 1
            if max_steps is not None and step >= max_steps:
                print(f"[INFO]: Reached --max-steps={max_steps}; exiting.")
                break

            # Hold the control rate to match the dataset's recorded fps
            # so the policy sees the temporal distribution it trained on.
            elapsed = time.perf_counter() - loop_start
            sleep_for = target_dt - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            if args.profile and step % max(args.profile_interval, 1) == 0:
                hz = 1.0 / (time.perf_counter() - loop_start)
                print(f"[PROFILE]: step={step} effective_rate={hz:.1f} Hz")
    except KeyboardInterrupt:
        print("\n[INFO]: Ctrl+C received; shutting down.")
    finally:
        if follower is not None:
            try:
                follower.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN]: follower.disconnect() raised: {exc}")
        if cameras is not None:
            for cam in cameras.values():
                try:
                    cam.disconnect()
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN]: camera.disconnect() raised: {exc}")
    return 0


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------


def _connect_so101_follower(*, port: str, robot_id: str):
    """Build + connect a LeRobot SO101 follower over the Feetech bus."""
    try:
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so101_follower import SO101FollowerConfig
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot's SO101 follower support is required. Install it via "
            "`bash scripts/install.sh` from the repo root."
        ) from exc

    follower = make_robot_from_config(
        SO101FollowerConfig(port=port, id=robot_id)
    )
    follower.connect()
    return follower


def _open_cameras(
    *,
    wrist_index: int,
    overhead_index: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Open the two USB cameras LeRobot expects in our dataset schema.

    Uses LeRobot's OpenCVCamera wrapper because that is what the
    upstream training datasets are recorded with — same backend, same
    color-space conventions, same uint8 RGB layout.
    """
    try:
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot's OpenCV camera support is required. Install LeRobot "
            "via `bash scripts/install.sh`."
        ) from exc

    cameras: dict[str, Any] = {}
    for name, index in (
        ("wrist_camera", int(wrist_index)),
        ("overhead_camera", int(overhead_index)),
    ):
        cam = OpenCVCamera(
            OpenCVCameraConfig(
                index_or_path=index,
                width=int(width),
                height=int(height),
                fps=30,
            )
        )
        cam.connect()
        cameras[name] = cam
    return cameras


# ---------------------------------------------------------------------------
# Policy + observation plumbing
# ---------------------------------------------------------------------------


def _load_lerobot_policy(checkpoint_path: str, *, device: str):
    """Resolve a checkpoint path (or its parent) to a PreTrainedPolicy.

    Delegates to `openso101.il.policies.load_policy` so sim playback and
    real deploy use exactly the same loader — same path resolution, same
    `PreTrainedConfig` → `get_policy_class` dispatch.
    """
    from openso101.il.policies import load_policy

    return load_policy(checkpoint_path, device=device)


def _build_real_observation(follower, cameras: Mapping[str, Any]) -> dict:
    """Build the per-step observation dict in the dataset's schema.

    Joint state is read straight from the follower in LeRobot motor
    units; camera frames are captured uint8 and converted to the same
    ``(1, 3, H, W)`` float-in-[0,1] tensor layout used by ``il play``.
    Keeping the schemas identical means policies are agnostic to the
    sim-vs-real source of observations.
    """
    import torch

    raw = follower.get_observation()
    # raw is a dict of ``"<joint>.pos": float`` plus possibly camera
    # frames if the follower was configured with cameras. We use the
    # joint values directly (no conversion) since the dataset records
    # them in the same motor-unit space.
    qpos_motor = _lerobot_joint_dict_to_array(raw)

    obs: dict = {
        "observation.state": torch.from_numpy(qpos_motor).unsqueeze(0).float(),
    }
    for cam_name, cam in cameras.items():
        frame = cam.read()  # H, W, 3 uint8
        tensor = (
            torch.from_numpy(np.asarray(frame, dtype=np.uint8))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        obs[f"observation.images.{cam_name}"] = tensor
    return obs


# ---------------------------------------------------------------------------
# LeRobot ↔ NumPy helpers (kept private to this module to avoid
# polluting the public teleop API surface)
# ---------------------------------------------------------------------------


_LEROBOT_JOINT_KEYS: tuple[str, ...] = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def _lerobot_joint_dict_to_array(raw: Mapping[str, float]) -> np.ndarray:
    """Extract joint positions in the canonical 6-dim order."""
    return np.asarray(
        [float(raw[key]) for key in _LEROBOT_JOINT_KEYS],
        dtype=np.float32,
    )


def _action_array_to_lerobot_dict(action: np.ndarray) -> dict[str, float]:
    """Inverse of :func:`_lerobot_joint_dict_to_array`."""
    if action.shape[-1] != len(_LEROBOT_JOINT_KEYS):
        raise ValueError(
            f"Policy action has shape {tuple(action.shape)}; expected last "
            f"dim={len(_LEROBOT_JOINT_KEYS)} matching LeRobot joint order."
        )
    return {key: float(value) for key, value in zip(_LEROBOT_JOINT_KEYS, action.tolist(), strict=True)}


__all__ = ["deploy"]
