# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


# Camera names must match the dataset schema produced by
# ``OpenSO101HDF5TeleopRecorder`` so a policy trained on teleop data
# can be deployed on hardware without renaming inputs.
_REAL_CAMERA_NAMES: tuple[str, ...] = ("wrist_camera", "overhead_camera")


def deploy(args: argparse.Namespace) -> int:
    """检查模型、设备、频率和停止文件，使用双相机与实际关节状态执行策略。"""
    _validate_camera_arguments(args)
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps 必须为正数")
    if args.profile_interval <= 0:
        raise ValueError("profile_interval 必须为正数")
    for name in ("wrist", "overhead"):
        source = _camera_source(args, name)
        if isinstance(source, Path) and source.is_file():
            raise ValueError("机器人部署需要实际相机设备")
    stop_file = Path(args.stop_file).expanduser().resolve() if args.stop_file else None
    if _stop_requested(stop_file):
        print("[INFO]: 停止文件存在，部署已终止。")
        return 0
    from openso101.il.runtime import resolve_policy_path
    from openso101.rl.gpu_guard import launch_cuda_command

    args.policy_path = str(resolve_policy_path(args.policy_path))
    launch_cuda_command(args.device)
    policy = _load_lerobot_policy(args.policy_path, device=args.device)
    goal_file = Path(args.goal_file).expanduser().resolve() if getattr(args, "goal_file", None) else None
    if getattr(policy, "metadata", {}).get("goal_input") == "robot_root_xyz_m":
        if goal_file is None:
            raise ValueError("student 部署需要 --goal-file 指定当前任务目标")
        _read_student_goal(goal_file)
    if hasattr(policy, "metadata") and "control_dt" in policy.metadata:
        if not np.isclose(1.0 / args.fps, policy.metadata["control_dt"], atol=1e-6):
            raise ValueError("部署 fps 必须与 student 训练控制频率一致")
    preprocessor = getattr(policy, "openso101_preprocessor", None)
    postprocessor = getattr(policy, "openso101_postprocessor", None)
    if preprocessor is None or postprocessor is None:
        raise RuntimeError("部署需要 checkpoint 的 LeRobot preprocessor 与 postprocessor")
    reset_action_dict = _reset_posture_action_dict()
    follower, cameras = None, None
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
            wrist_index=_camera_source(args, "wrist"),
            overhead_index=_camera_source(args, "overhead"),
            width=args.camera_width,
            height=args.camera_height,
            fps=args.fps,
        )
        print(f"[INFO]: Opened cameras: {list(cameras)}.")

        print(f"[INFO]: Loaded policy from {args.policy_path}.")
        if hasattr(policy, "reset"):
            policy.reset()

        policy_device = args.device

        # Startup safety: drive the follower to the canonical reset posture
        # and pause briefly before live control. Starting inference from an
        # arbitrary power-on pose can command a large first-step jump.
        if _stop_requested(stop_file):
            print("[INFO]: 停止文件存在，部署已终止。")
            return 0
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
            if _stop_requested(stop_file):
                print("[INFO]: 停止文件存在，部署已终止。")
                break

            obs = _build_real_observation(follower, cameras)
            if goal_file is not None:
                import torch

                obs["observation.goal"] = torch.as_tensor(_read_student_goal(goal_file), dtype=torch.float32).unsqueeze(0)
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

            if _stop_requested(stop_file):
                print("[INFO]: 停止文件存在，部署已终止。")
                break
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
        with ExitStack() as cleanup:
            if cameras is not None:
                for cam in cameras.values():
                    cleanup.callback(_disconnect_camera, cam)
            if follower is not None and follower.is_connected:
                cleanup.callback(follower.disconnect)
    return 0


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------


def _connect_so101_follower(*, port: str, robot_id: str):
    """Build + connect a LeRobot SO101 follower over the Feetech bus."""
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so101_follower import SO101FollowerConfig

    follower = make_robot_from_config(
        SO101FollowerConfig(port=port, id=robot_id)
    )
    with ExitStack() as cleanup:
        cleanup.callback(_disconnect_follower, follower)
        follower.connect()
        cleanup.pop_all()
    return follower


def _disconnect_follower(follower) -> None:
    if follower.is_connected:
        follower.disconnect()


def _open_cameras(
    *,
    wrist_index: int | Path,
    overhead_index: int | Path,
    width: int,
    height: int,
    fps: int = 30,
) -> dict[str, Any]:
    """Open the two USB cameras LeRobot expects in our dataset schema.

    Uses LeRobot's OpenCVCamera wrapper because that is what the
    upstream training datasets are recorded with — same backend, same
    color-space conventions, same uint8 RGB layout.
    """
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    cameras: dict[str, Any] = {}
    with ExitStack() as cleanup:
        for name, index in (
            ("wrist_camera", wrist_index),
            ("overhead_camera", overhead_index),
        ):
            video_file = isinstance(index, Path) and index.is_file()
            cam = OpenCVCamera(
                OpenCVCameraConfig(
                    index_or_path=index,
                    width=None if video_file else int(width),
                    height=None if video_file else int(height),
                    fps=None if video_file else int(fps),
                )
            )
            cleanup.callback(_disconnect_camera, cam)
            # 视频输入从首帧开始；设备输入执行 LeRobot warmup。
            cam.connect(warmup=not video_file)
            if cam.width != width or cam.height != height or not np.isclose(cam.fps, fps, atol=1e-3):
                raise ValueError(f"相机 metadata 与请求尺寸或 FPS 不一致: {name}")
            cameras[name] = cam
        cleanup.pop_all()
    return cameras


def _disconnect_camera(camera) -> None:
    if camera.is_connected:
        camera.disconnect()


def _camera_source(args, name: str) -> int | Path:
    path = getattr(args, f"{name}_camera_path", None)
    return Path(path).expanduser().resolve() if path is not None else getattr(args, f"{name}_camera_index")


def _validate_camera_arguments(args) -> None:
    if args.fps <= 0 or args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("fps 与相机尺寸必须为正数")
    if _camera_source(args, "wrist") == _camera_source(args, "overhead"):
        raise ValueError("wrist 与 overhead 必须使用不同相机来源")


def _stop_requested(path: Path | None) -> bool:
    if path is None:
        return False
    if path.exists() and not path.is_file():
        raise ValueError(f"停止路径必须为文件: {path}")
    return path.is_file()


def _camera_frame_tensor(frame: np.ndarray):
    import torch

    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or min(frame.shape[:2]) <= 0:
        raise ValueError("相机帧必须为 uint8 H×W×3 RGB")
    return torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0


# ---------------------------------------------------------------------------
# Policy + observation plumbing
# ---------------------------------------------------------------------------


def _load_lerobot_policy(checkpoint_path: str, *, device: str):
    """通过共享入口读取模型、配置和 processors。"""
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


def _reset_posture_action_dict() -> dict[str, float]:
    """使用共享初始姿态生成 LeRobot motor-unit 动作。"""
    import torch

    from openso101.robots.so101.constants import SO101_CANONICAL_INIT_JOINT_POS, SO101_SIM_JOINT_NAMES
    from openso101.teleop.so101_mapping import batched_action_to_motor_units

    rad = torch.tensor(
        [float(SO101_CANONICAL_INIT_JOINT_POS[name]) for name in SO101_SIM_JOINT_NAMES],
        dtype=torch.float32,
    )
    motor = batched_action_to_motor_units(rad).cpu().numpy().reshape(-1)
    return _action_array_to_lerobot_dict(_clamp_motor_units(motor))


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
    out = np.asarray(action, dtype=np.float32)
    if out.ndim == 2 and out.shape[0] == 1:
        out = out[0]
    if out.shape != (len(_LEROBOT_JOINT_KEYS),):
        raise ValueError(
            f"motor-unit action must have shape ({len(_LEROBOT_JOINT_KEYS)},), got {out.shape}"
        )
    if not np.isfinite(out).all():
        raise ValueError("motor-unit action must contain finite values")
    out = out.copy()
    for i, key in enumerate(_LEROBOT_JOINT_KEYS):
        lo, hi = _MOTOR_UNIT_CLAMP[key]
        out[i] = float(min(max(out[i], lo), hi))
    return out


def _read_student_goal(path: Path):
    value = np.asarray(json.loads(path.read_text()), dtype=np.float32)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("student 目标文件必须包含三个有限数值，单位为米")
    return value


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
    if set(cameras) != set(_REAL_CAMERA_NAMES):
        raise ValueError("部署需要 wrist_camera 与 overhead_camera")
    for cam_name, cam in cameras.items():
        frame = cam.read()  # H, W, 3 uint8
        tensor = _camera_frame_tensor(frame)
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
    values = np.asarray(
        [float(raw[key]) for key in _LEROBOT_JOINT_KEYS],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError("机器人关节观测必须为有限数值")
    return values


def _action_array_to_lerobot_dict(action: np.ndarray) -> dict[str, float]:
    """Inverse of :func:`_lerobot_joint_dict_to_array`."""
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 2 and action.shape[0] == 1:
        action = action[0]
    if action.shape != (len(_LEROBOT_JOINT_KEYS),):
        raise ValueError(
            f"Policy action has shape {tuple(action.shape)}; expected last "
            f"dim={len(_LEROBOT_JOINT_KEYS)} matching LeRobot joint order."
        )
    if not np.isfinite(action).all():
        raise ValueError("Policy action must contain finite values")
    return {key: float(value) for key, value in zip(_LEROBOT_JOINT_KEYS, action.tolist(), strict=True)}


__all__ = ["deploy"]
