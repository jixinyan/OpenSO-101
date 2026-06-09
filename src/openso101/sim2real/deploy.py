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

    The inference loop mirrors ``openso101.cli.il._cmd_play`` EXACTLY for
    the normalization plumbing: the LeRobot pre-processor is applied to the
    observation BEFORE ``select_action`` and the post-processor to the
    action AFTER. Without them the policy sees / emits N(0, 1) normalized
    values and the arm barely moves. Unlike ``il play`` we do NOT run
    ``batched_motor_units_to_action`` here — the real follower's
    ``send_action`` expects motor units, and that inverse (motor -> sim
    radians) is only for the SIM env.
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
        if hasattr(policy, "reset"):
            policy.reset()

        # Pull the LeRobot pre/post-processor pipelines the loader stashed on
        # the policy (see openso101.il.policies.factory.load_policy). These
        # carry the dataset normalization stats. A real-robot session must
        # NEVER run unnormalized: unnormalized observations produce a policy
        # output near the normalized-action mean, which on hardware is a
        # near-stationary / drifting command — at best the arm sits still, at
        # worst it lurches to a calibration midpoint. Fail loudly instead.
        preprocessor = getattr(policy, "openso101_preprocessor", None)
        postprocessor = getattr(policy, "openso101_postprocessor", None)
        if preprocessor is None or postprocessor is None:
            raise RuntimeError(
                "Refusing to deploy on real hardware without LeRobot "
                "pre/post-processors (normalization stats). "
                f"preprocessor={'set' if preprocessor is not None else 'None'}, "
                f"postprocessor={'set' if postprocessor is not None else 'None'}. "
                "The checkpoint must contain policy_preprocessor.json + "
                "policy_postprocessor.json. Re-export the checkpoint or load "
                "with allow_unnormalized=True ONLY for an unpowered dry run."
            )

        policy_device = args.device

        # Startup safety: drive the follower to the canonical reset posture
        # and pause briefly before live control. Starting inference from an
        # arbitrary power-on pose can command a large first-step jump.
        reset_action_dict = _reset_posture_action_dict(follower)
        if reset_action_dict is not None:
            print(f"[INFO]: Commanding canonical reset posture: {reset_action_dict}")
            follower.send_action(reset_action_dict)
            # Let the servos settle at the reset pose before live control.
            time.sleep(float(getattr(args, "reset_settle_seconds", 1.5)))

        # First-order ease-in: blend the policy command toward the previous
        # commanded target for the first few steps so the arm does not snap
        # from the reset pose to the policy's first prediction. alpha rises
        # from ~0 to 1 over `ease_in_steps` control steps.
        ease_in_steps = int(getattr(args, "ease_in_steps", 5))
        prev_command = (
            np.asarray(
                [reset_action_dict[k] for k in _LEROBOT_JOINT_KEYS],
                dtype=np.float32,
            )
            if reset_action_dict is not None
            else None
        )

        target_dt = 1.0 / float(max(1, args.fps))
        step = 0
        max_steps = int(args.max_steps) if args.max_steps is not None else None

        while True:
            loop_start = time.perf_counter()

            obs = _build_real_observation(follower, cameras)
            # Move obs to the policy device, then apply the preprocessor
            # (normalization) BEFORE select_action — identical to il play.
            obs = {
                k: (v.to(policy_device) if hasattr(v, "to") else v)
                for k, v in obs.items()
            }
            action = _run_policy(policy, obs, preprocessor, postprocessor)
            # Policy + postprocessor return shape (1, 6) MOTOR UNITS (no
            # motor->radian inverse here — the real follower wants motor
            # units). Squeeze to a plain 6-vector keyed by LeRobot names.
            action_np = action.detach().cpu().numpy().reshape(-1).astype(np.float32)

            # First-order ease-in for the opening steps.
            if prev_command is not None and step < ease_in_steps:
                alpha = float(step + 1) / float(max(1, ease_in_steps))
                action_np = (1.0 - alpha) * prev_command + alpha * action_np

            # Clamp every commanded target to the calibrated motor-unit range
            # before sending so a bad prediction can't drive the servo past
            # its safe span.
            action_np = _clamp_motor_units(action_np)
            prev_command = action_np
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


def _run_policy(policy, obs: Mapping[str, Any], preprocessor, postprocessor):
    """Apply preprocessor -> select_action -> postprocessor.

    Factored out so the application order matches ``il play`` exactly and
    can be exercised in isolation. ``obs`` is already on the policy device.
    Both processors are required (the caller raises if either is None) so
    this function does not silently skip normalization.
    """
    import torch

    with torch.inference_mode():
        obs = preprocessor(obs)
        action = policy.select_action(obs)
        action = postprocessor(action)
    return action


def _reset_posture_action_dict(follower) -> dict[str, float] | None:
    """Return the canonical reset posture as a LeRobot motor-unit action dict.

    Prefers :data:`SO101_CANONICAL_INIT_JOINT_POS` (radians, keyed by sim
    USD joint names) mapped through the sim-radian -> motor-unit conversion.
    The sim joint order ``(Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll,
    Jaw)`` is 1:1 with both :data:`_LEROBOT_JOINT_KEYS` and the mapping's
    ``JOINT_ORDER``, so a positional remap is exact.

    Falls back to reading the follower's current observation (i.e. "stay
    where you are") if the canonical pose or the conversion is unavailable,
    so deploy never hard-fails on a missing optional dependency. Returns
    ``None`` only if no reset pose can be determined at all.
    """
    try:
        import torch

        from openso101.robots import SO101_SIM_JOINT_NAMES
        from openso101.robots.so101.so_arm101 import SO101_CANONICAL_INIT_JOINT_POS
        from openso101.teleop.so101_mapping import batched_action_to_motor_units

        rad = torch.tensor(
            [float(SO101_CANONICAL_INIT_JOINT_POS[name]) for name in SO101_SIM_JOINT_NAMES],
            dtype=torch.float32,
        )
        motor = batched_action_to_motor_units(rad).cpu().numpy().reshape(-1)
        motor = _clamp_motor_units(np.asarray(motor, dtype=np.float32))
        return _action_array_to_lerobot_dict(motor)
    except Exception as exc:  # noqa: BLE001 — fall back to current pose
        print(
            f"[WARN]: Could not build canonical reset posture ({exc}); "
            "falling back to the follower's current pose as the reset target."
        )
        try:
            raw = follower.get_observation()
            motor = _clamp_motor_units(_lerobot_joint_dict_to_array(raw))
            return _action_array_to_lerobot_dict(motor)
        except Exception as exc2:  # noqa: BLE001
            print(f"[WARN]: Could not read follower pose for reset: {exc2}.")
            return None


# Calibrated commanded-target range for the real follower, in LeRobot motor
# units. The Feetech driver normalizes each servo to [-100, 100] (gripper
# [0, 100]); clamping here guarantees a bad policy prediction can never drive
# a servo past its safe span. These are the driver-convention endpoints, NOT
# a per-joint mechanical calibration — the real follower's calibration JSON
# is the true source on hardware (see so101_mapping.py for the analogous
# discrepancy note). Tighten per-joint here if a specific arm needs it.
_MOTOR_UNIT_CLAMP: dict[str, tuple[float, float]] = {
    "shoulder_pan.pos": (-100.0, 100.0),
    "shoulder_lift.pos": (-100.0, 100.0),
    "elbow_flex.pos": (-100.0, 100.0),
    "wrist_flex.pos": (-100.0, 100.0),
    "wrist_roll.pos": (-100.0, 100.0),
    "gripper.pos": (0.0, 100.0),
}


def _clamp_motor_units(action: np.ndarray) -> np.ndarray:
    """Clamp each commanded motor-unit target to its calibrated safe range."""
    out = np.asarray(action, dtype=np.float32).copy()
    for i, key in enumerate(_LEROBOT_JOINT_KEYS):
        lo, hi = _MOTOR_UNIT_CLAMP[key]
        out[i] = float(min(max(out[i], lo), hi))
    return out


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
