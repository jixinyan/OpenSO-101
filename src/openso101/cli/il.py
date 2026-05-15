# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""`openso101 il ...` subcommands."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Shared helpers (record + replay)
# ---------------------------------------------------------------------------


def _launch_isaac_app(args: argparse.Namespace, enable_cameras: bool = True):
    """Construct AppLauncher with cameras enabled and return the simulation app handle.

    AppLauncher MUST run before any isaaclab/openso101.tasks import — its
    side-effects bootstrap the Omniverse kit/extensions that those imports
    require.
    """
    from isaaclab.app import AppLauncher

    # AppLauncher reads attributes off the args namespace (headless, device,
    # enable_cameras, etc.); inject anything we know we need.
    if enable_cameras:
        args.enable_cameras = True
    if not hasattr(args, "headless"):
        args.headless = False
    if not hasattr(args, "device"):
        args.device = "cuda:0"
    if not hasattr(args, "disable_fabric"):
        args.disable_fabric = False
    app_launcher = AppLauncher(args)
    return app_launcher.app


# ---------------------------------------------------------------------------
# record (teleop_agent)
# ---------------------------------------------------------------------------


class _TeleopKeyboard:
    """Best-effort Isaac app-window teleop key handler."""

    def __init__(self):
        self.checkpoint_recording = False
        self.resume_recording = False
        self.toggle_recording = False
        self.quit_without_saving = False
        self._sub_keyboard = None
        try:
            import carb
            import omni.appwindow

            self._keyboard_event_type = carb.input.KeyboardEventType
            self._window = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._keyboard = self._window.get_keyboard()
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(
                self._keyboard, self._on_keyboard_event
            )
        except Exception as exc:  # pragma: no cover - depends on Isaac UI runtime
            self._keyboard_event_type = None
            print(f"[WARN]: Keyboard controls disabled: {exc}")

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type != self._keyboard_event_type.KEY_PRESS:
            return False
        key_name = event.input.name.upper()
        if key_name == "C":
            self.checkpoint_recording = True
            print("[INFO]: Checkpoint requested.")
            return True
        if key_name == "R":
            self.resume_recording = True
            print("[INFO]: Resume requested.")
            return True
        if key_name == "Q":
            self.quit_without_saving = True
            print("[INFO]: Quit without saving requested.")
            return True
        if key_name == "S":
            self.toggle_recording = True
            return True
        return False

    def cleanup(self) -> None:
        if self._sub_keyboard is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None


@dataclass
class _TeleopSimCheckpoint:
    recorder_checkpoint: Any
    robot_joint_pos: Any | None = None
    robot_joint_vel: Any | None = None
    hold_joint_target: Any | None = None
    object_root_state: Any | None = None
    command_stage: Any | None = None
    command_goal_pos_b: Any | None = None
    command_goal_pos_w: Any | None = None
    command_cube_spawn_xy_b: Any | None = None


def _clone_value(value):
    if hasattr(value, "clone"):
        return value.clone()
    if hasattr(value, "copy"):
        return value.copy()
    return value


def _checkpoint_joint_target(robot, joint_pos, sim_joint_names):
    """Return the six-joint teleop target matching a checkpointed robot pose."""

    try:
        joint_names = list(getattr(robot, "joint_names", ()))
        if joint_names:
            indices = [joint_names.index(name) for name in sim_joint_names]
            selected = joint_pos[..., indices]
        else:
            selected = joint_pos

        if getattr(selected, "ndim", 0) == 2 and selected.shape[0] == 1:
            selected = selected[0]
        return _clone_value(selected)
    except Exception as exc:  # pragma: no cover - depends on Isaac articulation metadata
        print(f"[WARN]: Could not build checkpoint hold target: {exc}")
        return None


def _coerce_target_like(target, reference):
    if hasattr(target, "to") and hasattr(reference, "device"):
        kwargs = {"device": reference.device}
        if hasattr(reference, "dtype"):
            kwargs["dtype"] = reference.dtype
        return target.to(**kwargs)
    if hasattr(reference, "new_tensor"):
        return reference.new_tensor(target)
    return target


def _target_max_abs_error(a, b) -> float:
    try:
        diff = a - b
        abs_diff = diff.abs() if hasattr(diff, "abs") else abs(diff)
        max_error = abs_diff.max() if hasattr(abs_diff, "max") else max(abs_diff)
        return float(max_error.item() if hasattr(max_error, "item") else max_error)
    except TypeError:
        return max(abs(float(a_value) - float(b_value)) for a_value, b_value in zip(a, b, strict=True))


@dataclass
class _ResumeHoldState:
    targets: Any
    holding: bool
    released: bool
    error: float | None


class _TeleopResumeHold:
    """Hold restored sim commands until the real leader arm is moved back near them."""

    def __init__(self, release_threshold: float):
        self.release_threshold = float(release_threshold)
        self._target = None

    @property
    def active(self) -> bool:
        return self._target is not None

    def activate(self, target) -> None:
        self._target = _clone_value(target)
        print(
            "[INFO]: Holding restored checkpoint pose. "
            "Move the real leader arm near the checkpoint pose to resume live control."
        )

    def apply(self, leader_targets) -> _ResumeHoldState:
        if self._target is None:
            return _ResumeHoldState(targets=leader_targets, holding=False, released=False, error=None)

        hold_target = _coerce_target_like(self._target, leader_targets)
        error = _target_max_abs_error(leader_targets, hold_target)
        if error <= self.release_threshold:
            self._target = None
            print(
                f"[INFO]: Leader arm re-synced to checkpoint pose; live control resumed (max error {error:.4f} rad)."
            )
            return _ResumeHoldState(targets=leader_targets, holding=False, released=True, error=error)

        return _ResumeHoldState(targets=hold_target, holding=True, released=False, error=error)


class _TeleopTargetRateLimiter:
    """Limit per-step target jumps so teleop does not inject unrealistic velocity spikes."""

    def __init__(self, max_delta: float):
        self.max_delta = float(max_delta)
        self._previous = None

    @property
    def enabled(self) -> bool:
        return self.max_delta > 0.0

    def reset(self, target=None) -> None:
        self._previous = _clone_value(target) if target is not None else None

    def apply(self, target):
        if not self.enabled:
            return target
        if self._previous is None:
            self._previous = _clone_value(target)
            return target

        previous = _coerce_target_like(self._previous, target)
        try:
            delta = target - previous
            if hasattr(delta, "clamp"):
                limited = previous + delta.clamp(min=-self.max_delta, max=self.max_delta)
            else:
                limited = previous + delta.clip(-self.max_delta, self.max_delta)
        except (AttributeError, TypeError):
            limited = [
                float(prev_value)
                + max(min(float(target_value) - float(prev_value), self.max_delta), -self.max_delta)
                for target_value, prev_value in zip(target, previous, strict=True)
            ]
        self._previous = _clone_value(limited)
        return limited


class _TeleopCheckpointStore:
    def __init__(self, env=None, scene=None, sim_joint_names=()):
        self.env = env
        self.scene = scene
        self.sim_joint_names = tuple(sim_joint_names)
        self.checkpoint: _TeleopSimCheckpoint | None = None

    @property
    def has_checkpoint(self) -> bool:
        return self.checkpoint is not None

    def capture(self, recorder) -> None:
        recorder_checkpoint = recorder.create_checkpoint() if hasattr(recorder, "create_checkpoint") else None
        checkpoint = _TeleopSimCheckpoint(recorder_checkpoint=recorder_checkpoint)

        if self.scene is not None:
            try:
                robot = self.scene["robot"]
                checkpoint.robot_joint_pos = robot.data.joint_pos.clone()
                checkpoint.robot_joint_vel = robot.data.joint_vel.clone()
                checkpoint.hold_joint_target = _checkpoint_joint_target(
                    robot, checkpoint.robot_joint_pos, self.sim_joint_names
                )
            except Exception as exc:  # pragma: no cover - depends on Isaac runtime
                print(f"[WARN]: Could not checkpoint robot state: {exc}")
            try:
                obj = self.scene["object"]
                checkpoint.object_root_state = obj.data.root_state_w.clone()
            except Exception as exc:  # pragma: no cover - depends on Isaac runtime
                print(f"[WARN]: Could not checkpoint object state: {exc}")

        if self.env is not None:
            try:
                command = self.env.command_manager.get_term("object_pose")
                checkpoint.command_stage = command.stage.clone()
                checkpoint.command_goal_pos_b = command.goal_pos_b.clone()
                checkpoint.command_goal_pos_w = command.goal_pos_w.clone()
                checkpoint.command_cube_spawn_xy_b = command.cube_spawn_xy_b.clone()
            except Exception:
                pass

        self.checkpoint = checkpoint
        print("[INFO]: Captured teleop checkpoint.")

    def restore(self, recorder):
        if self.checkpoint is None:
            return None
        checkpoint = self.checkpoint
        if checkpoint.recorder_checkpoint is not None and hasattr(recorder, "restore_checkpoint"):
            recorder.restore_checkpoint(checkpoint.recorder_checkpoint)

        if self.scene is not None:
            try:
                robot = self.scene["robot"]
                if checkpoint.robot_joint_pos is not None:
                    robot.write_joint_position_to_sim(checkpoint.robot_joint_pos)
                    robot.set_joint_position_target(checkpoint.robot_joint_pos)
                if checkpoint.robot_joint_vel is not None:
                    robot.write_joint_velocity_to_sim(checkpoint.robot_joint_vel)
            except Exception as exc:  # pragma: no cover - depends on Isaac runtime
                print(f"[WARN]: Could not restore robot state: {exc}")
            try:
                obj = self.scene["object"]
                if checkpoint.object_root_state is not None:
                    obj.write_root_state_to_sim(checkpoint.object_root_state)
            except Exception as exc:  # pragma: no cover - depends on Isaac runtime
                print(f"[WARN]: Could not restore object state: {exc}")

        if self.env is not None:
            try:
                command = self.env.command_manager.get_term("object_pose")
                if checkpoint.command_stage is not None:
                    command.stage[:] = checkpoint.command_stage
                if checkpoint.command_goal_pos_b is not None:
                    command.goal_pos_b[:] = checkpoint.command_goal_pos_b
                if checkpoint.command_goal_pos_w is not None:
                    command.goal_pos_w[:] = checkpoint.command_goal_pos_w
                if checkpoint.command_cube_spawn_xy_b is not None:
                    command.cube_spawn_xy_b[:] = checkpoint.command_cube_spawn_xy_b
            except Exception:
                pass
        print("[INFO]: Restored teleop checkpoint.")
        return checkpoint.hold_joint_target


def _handle_recording_key_events(keyboard, recorder, checkpoints=None, resume_hold=None) -> bool:
    """Apply pending recording key events and return whether teleop should exit."""

    should_quit = False
    if keyboard.quit_without_saving:
        keyboard.quit_without_saving = False
        if recorder is not None and recorder.recording:
            recorder.cancel_episode()
        should_quit = True

    if keyboard.checkpoint_recording:
        keyboard.checkpoint_recording = False
        if recorder is None:
            print("[WARN]: Checkpoint requested, but --repo-id was not provided.")
        elif recorder.recording:
            if checkpoints is not None:
                checkpoints.capture(recorder)
            elif hasattr(recorder, "create_checkpoint"):
                recorder.create_checkpoint()
            else:
                print("[WARN]: Checkpoints are not supported by the current recorder.")
        else:
            print("[WARN]: Checkpoint requested, but no episode is currently recording.")

    if keyboard.resume_recording:
        keyboard.resume_recording = False
        if recorder is None:
            print("[WARN]: Resume requested, but --repo-id was not provided.")
        elif checkpoints is not None and checkpoints.has_checkpoint:
            hold_target = checkpoints.restore(recorder)
            if resume_hold is not None and hold_target is not None:
                resume_hold.activate(hold_target)
        elif recorder.recording:
            print("[WARN]: Resume requested, but no checkpoint has been captured yet.")
        else:
            recorder.start_episode()

    if keyboard.toggle_recording:
        keyboard.toggle_recording = False
        if recorder is None:
            print("[WARN]: Recording requested, but --repo-id was not provided.")
        elif recorder.recording:
            recorder.save_episode()
        else:
            recorder.start_episode()

    return should_quit


def _success_prompt_response(response: str) -> bool:
    return response.strip().lower() in {"y", "yes"}


def _prompt_save_successful_episode(recorder, input_fn: Callable[[str], str] = input) -> None:
    if recorder is None or not recorder.recording:
        return
    try:
        response = input_fn("[SUCCESS]: Goal reached. Save this episode? [y/N]: ")
    except EOFError:
        response = ""
    if _success_prompt_response(response):
        recorder.save_episode(success=True)
    else:
        recorder.cancel_episode()


def _teleop_goal_success(env, command_name: str = "object_pose") -> bool:
    """Return true when the teleop object reaches the final pick/place goal."""

    try:
        command = env.command_manager.get_term(command_name)
        stage = command.stage
        if not bool((stage[0] >= 2).item()):
            return False

        from isaaclab.utils.math import subtract_frame_transforms

        cube_pos_b, _ = subtract_frame_transforms(
            command.robot.data.root_pos_w,
            command.robot.data.root_quat_w,
            command.object.data.root_pos_w,
        )
        final_goal_b = command.goal_for_stage(2)
        return bool(command.is_touching_goal(cube_pos_b, final_goal_b)[0].item())
    except Exception:
        return False


def _copy_targets_to_actions(actions, targets) -> None:
    if actions.ndim == 1:
        if actions.shape[0] != targets.shape[0]:
            raise RuntimeError(
                f"Action shape {tuple(actions.shape)} is incompatible with SO101 targets {tuple(targets.shape)}"
            )
        actions.copy_(targets)
        return

    if actions.ndim == 2 and actions.shape[-1] == targets.shape[0]:
        actions[:] = targets
        return

    raise RuntimeError(
        f"Action shape {tuple(actions.shape)} is incompatible with SO101 targets {tuple(targets.shape)}"
    )


def _tensor_to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return value


def _collect_replay_sim_state(unwrapped_env, scene) -> dict[str, Any]:
    """Collect optional simulator state that lets HDF5 teleop frames be replayed from checkpoints."""

    sim_state: dict[str, Any] = {}
    try:
        sim_state["object_root_state"] = _tensor_to_numpy(scene["object"].data.root_state_w[0])
    except Exception:
        pass
    try:
        command = unwrapped_env.command_manager.get_term("object_pose")
        sim_state["command_stage"] = _tensor_to_numpy(command.stage[0])
        sim_state["command_goal_pos_b"] = _tensor_to_numpy(command.goal_pos_b[0])
        sim_state["command_goal_pos_w"] = _tensor_to_numpy(command.goal_pos_w[0])
        sim_state["command_cube_spawn_xy_b"] = _tensor_to_numpy(command.cube_spawn_xy_b[0])
    except Exception:
        pass
    return sim_state


def _cmd_record(args: argparse.Namespace) -> int:
    # Fill in legacy defaults for flags not exposed by the new CLI.
    args.record_format = getattr(args, "record_format", "hdf5")
    args.print_actions = getattr(args, "print_actions", False)
    args.profile_interval = getattr(args, "profile_interval", 60)
    args.profile_joints = getattr(args, "profile_joints", False)
    args.goal_region = getattr(args, "goal_region", False)
    args.invert_joints = getattr(args, "invert_joints", "")
    args.joint_offsets_deg = getattr(args, "joint_offsets_deg", "")
    args.resume_sync_threshold = getattr(args, "resume_sync_threshold", 0.08)
    args.action_rate_limit = getattr(args, "action_rate_limit", 0.0)
    if args.repo_id is None:
        args.repo_id = "local/openso101_pickplace_teleop"
    if args.repo_root is None:
        args.repo_root = "./teleop_data/openso101_pickplace_teleop"
    if args.task_name is None:
        args.task_name = "Pick up the green cube and place it at the goal"

    simulation_app = _launch_isaac_app(args, enable_cameras=True)

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    import openso101.tasks  # noqa: F401
    from openso101.teleop.camera_viewports import open_teleop_viewports
    from openso101.teleop.hdf5_recorder import SafeSim2RealHDF5TeleopRecorder
    from openso101.teleop.lerobot_interface import LeRobotSO101Leader
    from openso101.teleop.lerobot_recorder import (
        SafeSim2RealLeRobotRecorder,
        collect_camera_buffers,
        discover_camera_metadata,
        ordered_action_to_numpy,
        read_robot_proprio,
    )
    from openso101.teleop.so101_mapping import (
        SO101_TELEOP_CONTROL_JOINT_NAMES,
        get_sim_joint_names,
        parse_joint_name_set,
        parse_joint_offsets_deg,
    )

    keyboard = _TeleopKeyboard()
    env = None
    recorder = None
    startup_error = None
    try:
        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=not args.disable_fabric,
        )
        if args.num_envs is None and hasattr(env_cfg, "scene"):
            env_cfg.scene.num_envs = 1
        try:
            env_cfg.scene.ee_frame.debug_vis = False
        except AttributeError:
            pass
        try:
            env_cfg.commands.object_pose.debug_vis = bool(args.goal_region)
        except AttributeError:
            pass

        inverted_joints = parse_joint_name_set(args.invert_joints)
        joint_offsets_rad = parse_joint_offsets_deg(args.joint_offsets_deg)
        sim_joint_names = get_sim_joint_names()
        if inverted_joints or joint_offsets_rad:
            print(
                "[INFO]: Teleop mapping calibration: "
                f"invert={sorted(inverted_joints)} offsets_rad={joint_offsets_rad}"
            )
        print(f"[INFO]: Teleop sim joints: {list(sim_joint_names)}")

        env = gym.make(args.task, cfg=env_cfg)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        print("[INFO]: Teleop keys: C checkpoint, R restore checkpoint, Q quit without saving, S save/start.")

        scene = env.unwrapped.scene
        unwrapped_env = env.unwrapped
        camera_metadata = discover_camera_metadata(scene)
        print(f"[INFO]: Recording cameras: {', '.join(camera_metadata)}")

        checkpoints = None
        if not args.no_record:
            if args.record_format == "hdf5":
                recorder = SafeSim2RealHDF5TeleopRecorder(
                    root=args.repo_root,
                    task_name=args.task_name,
                    cameras=camera_metadata,
                    fps=args.fps,
                    dataset_id=args.repo_id,
                    sim_joint_names=sim_joint_names,
                )
            else:
                recorder = SafeSim2RealLeRobotRecorder(
                    repo_id=args.repo_id,
                    root=args.repo_root,
                    task_name=args.task_name,
                    cameras=camera_metadata,
                    fps=args.fps,
                )
            recorder.init_dataset()
            recorder.start_episode()
            checkpoints = _TeleopCheckpointStore(
                env=unwrapped_env, scene=scene, sim_joint_names=sim_joint_names
            )
            print(
                f"[INFO]: Recording is local-only ({args.record_format}). Dataset root: {args.repo_root}"
            )
            print("[INFO]: Recording started automatically. Goal success prompts for save/discard.")
            print("[INFO]: Recording keys: C checkpoint, R restore checkpoint, Q quit without saving, S save/start.")

        leader = LeRobotSO101Leader(
            port=args.leader_port,
            robot_id=args.leader_id,
            inverted_joints=inverted_joints,
            joint_offsets_rad=joint_offsets_rad,
        )
        leader.connect()
        print(f"[INFO]: Connected SO101 leader '{args.leader_id}' on {args.leader_port}.")

        env.reset()
        if not args.no_camera_viewports:
            open_teleop_viewports(scene)
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        resume_hold = _TeleopResumeHold(args.resume_sync_threshold)
        target_limiter = _TeleopTargetRateLimiter(args.action_rate_limit)
        control_step = 0

        while simulation_app.is_running():
            with torch.inference_mode():
                loop_start = time.perf_counter()
                read_start = time.perf_counter()
                raw_action, targets = leader.read_target_tensor(env.unwrapped.device)
                read_ms = (time.perf_counter() - read_start) * 1000.0
                if args.print_actions:
                    print(raw_action)
                hold_state = resume_hold.apply(targets)
                applied_targets = hold_state.targets
                if hold_state.released:
                    target_limiter.reset(applied_targets)
                applied_targets = target_limiter.apply(applied_targets)
                _copy_targets_to_actions(actions, applied_targets)
                step_start = time.perf_counter()
                env.step(actions)
                step_ms = (time.perf_counter() - step_start) * 1000.0
                target_np = ordered_action_to_numpy(applied_targets)
                qpos, qvel = read_robot_proprio(scene["robot"], sim_joint_names=sim_joint_names)

                if recorder is not None and recorder.recording:
                    record_start = time.perf_counter()
                    recorder.add_frame(
                        action=target_np,
                        qpos=qpos,
                        qvel=qvel,
                        camera_buffers=collect_camera_buffers(scene),
                        timestamp=time.time(),
                        sim_state=_collect_replay_sim_state(unwrapped_env, scene),
                    )
                    record_ms = (time.perf_counter() - record_start) * 1000.0
                else:
                    record_ms = 0.0

                if args.profile_teleop and control_step % max(args.profile_interval, 1) == 0:
                    joint_error = abs(qpos - target_np)
                    loop_ms = (time.perf_counter() - loop_start) * 1000.0
                    print(
                        "[PROFILE]: "
                        f"read={read_ms:.1f}ms step={step_ms:.1f}ms record={record_ms:.1f}ms "
                        f"loop={loop_ms:.1f}ms joint_error_mean={joint_error.mean():.4f}rad "
                        f"joint_error_max={joint_error.max():.4f}rad"
                    )
                    if hold_state.holding and hold_state.error is not None:
                        print(
                            "[PROFILE]: "
                            f"checkpoint_hold_active=1 leader_checkpoint_error_max={hold_state.error:.4f}rad "
                            f"release_threshold={resume_hold.release_threshold:.4f}rad"
                        )
                    if args.profile_joints:
                        joint_report = " ".join(
                            f"{name}:target={target:.3f},pos={pos:.3f},err={err:.3f}"
                            for name, target, pos, err in zip(
                                SO101_TELEOP_CONTROL_JOINT_NAMES,
                                target_np,
                                qpos,
                                joint_error,
                                strict=True,
                            )
                        )
                        raw_report = " ".join(
                            f"{name}={float(raw_action[f'{name}.pos']):.1f}"
                            for name in SO101_TELEOP_CONTROL_JOINT_NAMES
                        )
                        print(f"[PROFILE_JOINTS]: raw=({raw_report}) sim=({joint_report})")
                control_step += 1

                if args.goal_region and _teleop_goal_success(unwrapped_env):
                    _prompt_save_successful_episode(recorder)
                    break
                if _handle_recording_key_events(keyboard, recorder, checkpoints, resume_hold):
                    break
    except Exception as exc:
        startup_error = exc
        print("[ERROR]: Teleop agent failed before clean shutdown:", flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    finally:
        if recorder is not None and recorder.recording:
            recorder.save_episode()
        keyboard.cleanup()
        if env is not None:
            env.close()
        if startup_error is None:
            simulation_app.close()

    if startup_error is not None:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    return 0


# ---------------------------------------------------------------------------
# push (push_dataset)
# ---------------------------------------------------------------------------


def _push_hdf5_radians_to_motor_units(values):
    """Remap a sim-radian 6-vector from HDF5 to LeRobot STS3215 motor units."""
    import numpy as np
    import torch

    from openso101.teleop.so101_mapping import batched_action_to_motor_units

    tensor = torch.as_tensor(np.asarray(values, dtype=np.float32))
    return batched_action_to_motor_units(tensor).numpy().astype(np.float32, copy=False)


def _push_relative_missing_lerobot_paths(root: Path) -> list[str]:
    required_paths = (Path("meta/info.json"), Path("meta/tasks.parquet"), Path("meta/stats.json"))
    return [str(relative_path) for relative_path in required_paths if not (root / relative_path).is_file()]


def _push_lerobot_episode_files(root: Path) -> list[Path]:
    data_root = root / "data"
    if not data_root.is_dir():
        return []
    return sorted(data_root.glob("**/*.parquet"))


def _push_detect_input_format(root: Path) -> str:
    if (root / "episodes").is_dir() and list((root / "episodes").glob("episode_*.hdf5")):
        return "hdf5"
    if (root / "meta").is_dir() or (root / "data").is_dir():
        return "lerobot"
    raise SystemExit(
        f"Cannot detect dataset format for {root}. Expected local HDF5 episodes under 'episodes/' "
        "or a local LeRobot dataset with 'meta/' and 'data/'."
    )


def _push_camera_metadata_from_hdf5_episode(episode_file: Path) -> dict[str, dict[str, int]]:
    import h5py

    with h5py.File(episode_file, "r") as h5:
        cameras: dict[str, dict[str, int]] = {}
        for camera_name in ("wrist_camera", "overhead_camera"):
            shape = h5[f"observations/images/{camera_name}"].shape
            if int(shape[1]) < 16 or int(shape[2]) < 16:
                raise SystemExit(
                    f"{episode_file} camera '{camera_name}' is {int(shape[1])}x{int(shape[2])}; "
                    "LeRobot video export requires camera frames at least 16x16."
                )
            cameras[camera_name] = {"height": int(shape[1]), "width": int(shape[2])}
        return cameras


def _push_features_from_hdf5_episode(episode_file: Path, fps: int) -> dict[str, dict]:
    from openso101.teleop.so101_mapping import (
        LEROBOT_SO101_ACTION_NAMES,
        SO101_TELEOP_CONTROL_JOINT_NAMES,
    )

    cameras = _push_camera_metadata_from_hdf5_episode(episode_file)
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "fps": fps,
            "shape": (len(SO101_TELEOP_CONTROL_JOINT_NAMES),),
            "names": list(LEROBOT_SO101_ACTION_NAMES),
        },
        "action": {
            "dtype": "float32",
            "fps": fps,
            "shape": (len(SO101_TELEOP_CONTROL_JOINT_NAMES),),
            "names": list(LEROBOT_SO101_ACTION_NAMES),
        },
    }
    for camera_name, camera in cameras.items():
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "fps": fps,
            "shape": (camera["height"], camera["width"], 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _push_archive_existing_export(root: Path) -> Path:
    archive = root.with_name(f"{root.name}.previous-export")
    suffix = 1
    while archive.exists():
        archive = root.with_name(f"{root.name}.previous-export-{suffix}")
        suffix += 1
    root.rename(archive)
    return archive


def _push_validate_local_dataset(root: Path, input_format: str = "auto") -> list[Path]:
    from openso101.teleop.hdf5_recorder import validate_hdf5_dataset
    from openso101.teleop.lerobot_recorder import has_lerobot_metadata

    if not root.exists():
        raise SystemExit(f"Local dataset root does not exist: {root}")
    resolved_format = _push_detect_input_format(root) if input_format == "auto" else input_format
    if resolved_format == "hdf5":
        try:
            episode_files = validate_hdf5_dataset(root)
            for episode_file in episode_files:
                _push_camera_metadata_from_hdf5_episode(episode_file)
            return episode_files
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if not has_lerobot_metadata(root):
        missing = ", ".join(_push_relative_missing_lerobot_paths(root))
        raise SystemExit(
            f"Local LeRobot dataset is not ready to push: {root}. "
            f"Missing LeRobot metadata: {missing}. Save/export at least one teleop episode before pushing."
        )
    episode_files = _push_lerobot_episode_files(root)
    if not episode_files:
        raise SystemExit(f"Local dataset has no recorded episode parquet files under {root / 'data'}.")
    return episode_files


def _push_convert_hdf5_to_lerobot(
    hdf5_root: Path,
    lerobot_root: Path,
    repo_id: str,
    overwrite_export: bool = False,
) -> Path:
    import h5py
    import numpy as np

    from openso101.teleop.hdf5_recorder import validate_hdf5_dataset

    episode_files = validate_hdf5_dataset(hdf5_root)
    if lerobot_root.exists():
        if not overwrite_export:
            raise SystemExit(
                f"LeRobot export root already exists: {lerobot_root}. "
                "Pass --overwrite-export to archive it and rebuild the export."
            )
        archive = _push_archive_existing_export(lerobot_root)
        print(f"[WARN]: Archived existing LeRobot export root to {archive}")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("LeRobot is required to convert HDF5 teleop data.") from exc

    with h5py.File(episode_files[0], "r") as first_episode:
        fps = int(first_episode.attrs.get("fps", 30))
    dataset = LeRobotDataset.create(
        repo_id,
        fps=fps,
        features=_push_features_from_hdf5_episode(episode_files[0], fps=fps),
        root=lerobot_root,
        robot_type="so101_follower",
    )

    for episode_file in episode_files:
        with h5py.File(episode_file, "r") as h5:
            task = h5.attrs.get("task", "OpenSO-101 teleoperation")
            frame_count = h5["action"].shape[0]
            for frame_idx in range(frame_count):
                action_rad = np.asarray(h5["action"][frame_idx], dtype=np.float32)
                qpos_rad = np.asarray(h5["observations/qpos"][frame_idx], dtype=np.float32)
                frame = {
                    "action": _push_hdf5_radians_to_motor_units(action_rad),
                    "observation.state": _push_hdf5_radians_to_motor_units(qpos_rad),
                    "task": task,
                }
                for camera_name in ("wrist_camera", "overhead_camera"):
                    frame[f"observation.images.{camera_name}"] = np.asarray(
                        h5[f"observations/images/{camera_name}"][frame_idx],
                        dtype=np.uint8,
                    )
                dataset.add_frame(frame)
            dataset.save_episode()
            print(f"[INFO]: Exported {episode_file.name} to local LeRobot dataset.")
    return lerobot_root


def _cmd_push(args: argparse.Namespace) -> int:
    # Legacy-default fields not exposed on the new CLI surface.
    input_format = getattr(args, "input_format", "auto")
    lerobot_root = getattr(args, "lerobot_root", None)
    tags = getattr(args, "tag", None) or ["openso101", "so101", "teleoperation"]
    allow_pattern = getattr(args, "allow_pattern", None)
    upload_large_folder = getattr(args, "upload_large_folder", False)

    repo_root = Path(args.repo_root).expanduser().resolve()
    resolved_input_format = (
        _push_detect_input_format(repo_root) if input_format == "auto" else input_format
    )
    episode_files = _push_validate_local_dataset(repo_root, input_format=resolved_input_format)

    print(f"[INFO]: Local dataset root: {repo_root}")
    print(f"[INFO]: Input format: {resolved_input_format}")
    print(f"[INFO]: Local episode files: {len(episode_files)}")
    print(f"[INFO]: Target Hub dataset: {args.repo_id}")
    if args.dry_run:
        print("[INFO]: Dry run only; no Hub upload performed.")
        return 0

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("LeRobot is required to push datasets from this script.") from exc

    if resolved_input_format == "hdf5":
        export_root = Path(lerobot_root) if lerobot_root is not None else (repo_root / "lerobot_dataset")
        repo_root = _push_convert_hdf5_to_lerobot(
            hdf5_root=repo_root,
            lerobot_root=export_root.expanduser().resolve(),
            repo_id=args.repo_id,
            overwrite_export=args.overwrite_export,
        )

    dataset = LeRobotDataset(args.repo_id, root=repo_root)
    dataset.push_to_hub(
        branch=args.branch,
        tags=tags,
        license=args.license,
        tag_version=not args.no_tag_version,
        push_videos=not args.no_videos,
        private=args.private,
        allow_patterns=allow_pattern,
        upload_large_folder=upload_large_folder,
    )
    print(f"[INFO]: Pushed local LeRobot dataset to https://huggingface.co/datasets/{args.repo_id}")
    return 0


# ---------------------------------------------------------------------------
# replay (replay_teleop_checkpoint)
# ---------------------------------------------------------------------------


def _replay_episode_files(root: Path) -> list[Path]:
    episodes_dir = root / "episodes"
    if not episodes_dir.is_dir():
        return []
    return sorted(episodes_dir.glob("episode_*.hdf5"))


def _replay_resolve_episode_path(
    repo_root: Path | None,
    episode_path: Path | None,
    episode_index: int = -1,
) -> Path:
    if episode_path is not None:
        resolved = episode_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Teleop episode does not exist: {resolved}")
        return resolved

    if repo_root is None:
        raise FileNotFoundError("Either --episode or a repo root must be provided.")
    root = repo_root.expanduser().resolve()
    episodes = _replay_episode_files(root)
    if not episodes:
        raise FileNotFoundError(f"No teleop HDF5 episodes found under {root / 'episodes'}")
    index = episode_index if episode_index >= 0 else len(episodes) + episode_index
    if index < 0 or index >= len(episodes):
        raise IndexError(f"Episode index {episode_index} is out of range for {len(episodes)} episode(s).")
    return episodes[index]


def _replay_read_checkpoint_frames(episode_path: Path):
    import h5py
    import numpy as np

    with h5py.File(episode_path, "r") as h5:
        if "checkpoints/frame_index" not in h5:
            return np.asarray([], dtype=np.int64)
        return np.asarray(h5["checkpoints/frame_index"][:], dtype=np.int64)


def _replay_select_checkpoint_frame(
    episode_path: Path,
    checkpoint_frame: int | None = None,
    checkpoint_index: int = -1,
) -> int:
    import h5py

    with h5py.File(episode_path, "r") as h5:
        frame_count = int(h5["action"].shape[0])
    if frame_count <= 0:
        raise ValueError(f"Episode has no action frames: {episode_path}")

    if checkpoint_frame is not None:
        frame = int(checkpoint_frame)
    else:
        checkpoint_frames = _replay_read_checkpoint_frames(episode_path)
        if len(checkpoint_frames) == 0:
            frame = 0
        else:
            index = checkpoint_index if checkpoint_index >= 0 else len(checkpoint_frames) + checkpoint_index
            if index < 0 or index >= len(checkpoint_frames):
                raise IndexError(
                    f"Checkpoint index {checkpoint_index} is out of range for {len(checkpoint_frames)} checkpoint(s)."
                )
            frame = int(checkpoint_frames[index])

    if frame < 0 or frame >= frame_count:
        raise IndexError(f"Checkpoint frame {frame} is out of range for {frame_count} frame(s).")
    return frame


def _replay_frame_range(
    frame_count: int,
    checkpoint_frame: int,
    start_frame: int | None,
    stop_frame: int | None,
    max_steps: int | None,
) -> range:
    start = checkpoint_frame if start_frame is None else int(start_frame)
    stop = frame_count if stop_frame is None else min(int(stop_frame), frame_count)
    if max_steps is not None:
        stop = min(stop, start + int(max_steps))
    if start < 0 or start > frame_count:
        raise IndexError(f"Start frame {start} is out of range for {frame_count} frame(s).")
    if stop < start:
        raise IndexError(f"Stop frame {stop} must be >= start frame {start}.")
    return range(start, stop)


def _replay_to_tensor_like(values, reference):
    import torch

    return torch.as_tensor(values, device=reference.device, dtype=reference.dtype)


def _replay_robot_joint_indices(robot) -> list[int]:
    from openso101.robots import SO101_SIM_JOINT_NAMES

    joint_names = list(robot.joint_names)
    return [joint_names.index(joint_name) for joint_name in SO101_SIM_JOINT_NAMES]


def _replay_set_robot_proprio(scene, qpos, qvel) -> None:
    robot = scene["robot"]
    joint_ids = _replay_robot_joint_indices(robot)
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    joint_pos[0, joint_ids] = _replay_to_tensor_like(qpos, joint_pos[0, joint_ids])
    joint_vel[0, joint_ids] = _replay_to_tensor_like(qvel, joint_vel[0, joint_ids])
    robot.write_joint_position_to_sim(joint_pos)
    robot.write_joint_velocity_to_sim(joint_vel)
    robot.set_joint_position_target(joint_pos)


def _replay_optional_frame(h5, dataset_name: str, frame_index: int):
    import numpy as np

    if dataset_name not in h5:
        return None
    return np.asarray(h5[dataset_name][frame_index])


def _replay_restore_sim_state_from_episode(unwrapped_env, scene, h5, frame_index: int) -> None:
    import numpy as np

    _replay_set_robot_proprio(
        scene,
        qpos=np.asarray(h5["observations/qpos"][frame_index], dtype=np.float32),
        qvel=np.asarray(h5["observations/qvel"][frame_index], dtype=np.float32),
    )

    object_root_state = _replay_optional_frame(h5, "sim/object_root_state", frame_index)
    if object_root_state is None:
        print(
            "[WARN]: Episode has no sim/object_root_state. Restored robot pose only; object state may not match checkpoint."
        )
    else:
        obj = scene["object"]
        root_state = _replay_to_tensor_like(object_root_state[None, ...], obj.data.root_state_w)
        obj.write_root_state_to_sim(root_state)

    try:
        command = unwrapped_env.command_manager.get_term("object_pose")
        command_values = {
            "command_stage": _replay_optional_frame(h5, "sim/command_stage", frame_index),
            "command_goal_pos_b": _replay_optional_frame(h5, "sim/command_goal_pos_b", frame_index),
            "command_goal_pos_w": _replay_optional_frame(h5, "sim/command_goal_pos_w", frame_index),
            "command_cube_spawn_xy_b": _replay_optional_frame(h5, "sim/command_cube_spawn_xy_b", frame_index),
        }
        if command_values["command_stage"] is not None:
            command.stage[0] = _replay_to_tensor_like(command_values["command_stage"], command.stage[0])
        if command_values["command_goal_pos_b"] is not None:
            command.goal_pos_b[0] = _replay_to_tensor_like(
                command_values["command_goal_pos_b"], command.goal_pos_b[0]
            )
        if command_values["command_goal_pos_w"] is not None:
            command.goal_pos_w[0] = _replay_to_tensor_like(
                command_values["command_goal_pos_w"], command.goal_pos_w[0]
            )
        if command_values["command_cube_spawn_xy_b"] is not None:
            command.cube_spawn_xy_b[0] = _replay_to_tensor_like(
                command_values["command_cube_spawn_xy_b"], command.cube_spawn_xy_b[0]
            )
    except Exception as exc:
        print(f"[WARN]: Could not restore command state from episode: {exc}")


def _replay_copy_targets_to_actions(actions, targets) -> None:
    if actions.ndim == 1:
        actions.copy_(targets)
    else:
        actions[:] = targets


def _replay_step_action(env, actions, target, real_time_dt: float | None = None) -> None:
    step_start = time.perf_counter()
    _replay_copy_targets_to_actions(
        actions,
        _replay_to_tensor_like(target, actions[0] if actions.ndim == 2 else actions),
    )
    env.step(actions)
    if real_time_dt is not None:
        elapsed = time.perf_counter() - step_start
        if elapsed < real_time_dt:
            time.sleep(real_time_dt - elapsed)


def _replay_print_checkpoints(episode_path: Path) -> None:
    checkpoint_frames = _replay_read_checkpoint_frames(episode_path)
    if len(checkpoint_frames) == 0:
        print(
            f"[INFO]: {episode_path} has no saved checkpoint frames. "
            "Use a raw frame index to replay from a non-checkpoint frame."
        )
        return
    for index, frame in enumerate(checkpoint_frames):
        print(f"{index}: frame {int(frame)}")


def _cmd_replay(args: argparse.Namespace) -> int:
    from openso101.teleop.hdf5_recorder import validate_hdf5_episode

    # Resolve the episode path. The new CLI exposes a single --episode flag
    # which is the explicit path; the legacy CLI also supported --repo-root /
    # --episode-index but those are not surfaced here.
    episode_path = _replay_resolve_episode_path(
        repo_root=Path(getattr(args, "repo_root", "")) if getattr(args, "repo_root", None) else None,
        episode_path=Path(args.episode) if args.episode else None,
        episode_index=getattr(args, "episode_index", -1),
    )
    validate_hdf5_episode(episode_path)

    if args.list_checkpoints:
        _replay_print_checkpoints(episode_path)
        return 0

    # Legacy defaults for flags not exposed on the new CLI surface.
    args.checkpoint_frame = getattr(args, "checkpoint_frame", None)
    args.checkpoint_index = getattr(args, "checkpoint_index", -1)
    args.warm_start = getattr(args, "warm_start", False)
    args.hold_steps = getattr(args, "hold_steps", 30)
    args.no_camera_viewports = getattr(args, "no_camera_viewports", False)
    if args.task is None:
        args.task = "OpenSO101-PickPlace-Teleop-Vision-v0"

    simulation_app = _launch_isaac_app(args, enable_cameras=True)

    import gymnasium as gym
    import h5py
    import numpy as np
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    import openso101.tasks  # noqa: F401
    from openso101.teleop.camera_viewports import open_teleop_viewports

    env = None
    try:
        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=not args.disable_fabric,
        )
        if args.num_envs is None and hasattr(env_cfg, "scene"):
            env_cfg.scene.num_envs = 1

        env = gym.make(args.task, cfg=env_cfg)
        scene = env.unwrapped.scene
        unwrapped_env = env.unwrapped
        env.reset()
        if not args.no_camera_viewports:
            open_teleop_viewports(scene)

        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        with h5py.File(episode_path, "r") as h5:
            frame_count = int(h5["action"].shape[0])
            checkpoint_frame = _replay_select_checkpoint_frame(
                episode_path, args.checkpoint_frame, args.checkpoint_index
            )
            fps = int(h5.attrs.get("fps", 30))
            real_time_dt = 1.0 / fps if args.real_time and fps > 0 else None

            print(f"[INFO]: Replaying teleop episode: {episode_path}")
            print(f"[INFO]: Checkpoint frame: {checkpoint_frame}")
            if args.warm_start:
                print(f"[INFO]: Warm-starting by stepping recorded actions 0:{checkpoint_frame}.")
                for frame_index in range(checkpoint_frame):
                    _replay_step_action(
                        env,
                        actions,
                        np.asarray(h5["action"][frame_index], dtype=np.float32),
                        real_time_dt,
                    )
            else:
                _replay_restore_sim_state_from_episode(unwrapped_env, scene, h5, checkpoint_frame)

            checkpoint_action = np.asarray(h5["action"][checkpoint_frame], dtype=np.float32)
            for _ in range(max(args.hold_steps, 0)):
                _replay_step_action(env, actions, checkpoint_action, real_time_dt)

            replay_range = _replay_frame_range(
                frame_count=frame_count,
                checkpoint_frame=checkpoint_frame,
                start_frame=args.start_frame,
                stop_frame=args.stop_frame,
                max_steps=args.max_steps,
            )
            print(
                f"[INFO]: Running recorded teleop actions for frames {replay_range.start}:{replay_range.stop}."
            )
            for frame_index in replay_range:
                if not simulation_app.is_running():
                    break
                _replay_step_action(
                    env,
                    actions,
                    np.asarray(h5["action"][frame_index], dtype=np.float32),
                    real_time_dt,
                )
    finally:
        if env is not None:
            env.close()
        simulation_app.close()
    return 0


# ---------------------------------------------------------------------------
# train / play (sub-project C; out of scope for this port)
# ---------------------------------------------------------------------------


def _cmd_train(args: argparse.Namespace) -> int:
    print(
        "openso101 il train: not implemented in this refactor. "
        "See sub-project C (docs/superpowers/specs/2026-05-13-openso101-refactor-design.md § 13)."
    )
    return 2


def _cmd_play(args: argparse.Namespace) -> int:
    print(
        "openso101 il play: not implemented in this refactor. "
        "See sub-project C."
    )
    return 2


# ---------------------------------------------------------------------------
# subparser registration
# ---------------------------------------------------------------------------


def add_subparsers(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="il_cmd", required=True)

    p_rec = sub.add_parser("record", help="Record teleop demonstrations")
    p_rec.add_argument("--task", required=True)
    p_rec.add_argument("--leader-port", required=True)
    p_rec.add_argument("--leader-id", required=True)
    p_rec.add_argument("--repo-id")
    p_rec.add_argument("--repo-root")
    p_rec.add_argument("--task-name")
    p_rec.add_argument("--num-envs", type=int, default=None)
    p_rec.add_argument("--fps", type=int, default=30)
    p_rec.add_argument("--no-record", action="store_true")
    p_rec.add_argument("--profile-teleop", action="store_true")
    p_rec.add_argument("--no-camera-viewports", action="store_true")
    p_rec.set_defaults(func=_cmd_record)

    p_push = sub.add_parser("push", help="Push HDF5 dataset to LeRobot Hub")
    p_push.add_argument("--repo-root", required=True)
    p_push.add_argument("--repo-id", required=True)
    p_push.add_argument("--branch", default=None)
    p_push.add_argument("--private", action="store_true")
    p_push.add_argument("--license", default="apache-2.0")
    p_push.add_argument("--no-tag-version", action="store_true")
    p_push.add_argument("--no-videos", action="store_true")
    p_push.add_argument("--dry-run", action="store_true")
    p_push.add_argument("--overwrite-export", action="store_true")
    p_push.set_defaults(func=_cmd_push)

    p_train = sub.add_parser("train", help="Train an IL policy")
    p_train.add_argument(
        "--policy", required=True, choices=["act", "diffusion"]
    )
    p_train.add_argument("--dataset", required=True)
    p_train.set_defaults(func=_cmd_train)

    p_play = sub.add_parser("play", help="Replay a trained IL policy")
    p_play.add_argument("--task", required=True)
    p_play.add_argument("--policy-path", required=True)
    p_play.set_defaults(func=_cmd_play)

    p_replay = sub.add_parser(
        "replay", help="Replay a recorded teleop episode in sim"
    )
    p_replay.add_argument("--episode", required=True)
    p_replay.add_argument("--task", default=None)
    p_replay.add_argument("--num-envs", type=int, default=None)
    p_replay.add_argument("--start-frame", type=int, default=None)
    p_replay.add_argument("--stop-frame", type=int, default=None)
    p_replay.add_argument("--max-steps", type=int, default=None)
    p_replay.add_argument("--real-time", action="store_true")
    p_replay.add_argument("--list-checkpoints", action="store_true")
    p_replay.set_defaults(func=_cmd_replay)
