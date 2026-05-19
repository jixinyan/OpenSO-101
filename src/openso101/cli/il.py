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
    """Best-effort Isaac app-window teleop key handler.

    Each key sets a request flag; the dispatcher
    :func:`_handle_recording_key_events` consumes flags once per loop
    iteration. Attribute names match what each key actually does today
    (see the docstring on :func:`_handle_recording_key_events`):

      * ``take_checkpoint`` — C
      * ``restore_checkpoint`` — R
      * ``mark_success`` — S (saves episode as SUCCESS and exits)
      * ``quit_discard`` — Q (cancels episode and exits)

    Legacy attribute names (``checkpoint_recording``, ``resume_recording``,
    ``toggle_recording``, ``quit_without_saving``) are aliased via
    properties so external callers / tests that imported the old names
    keep working without behavior change.
    """

    def __init__(self):
        self.take_checkpoint = False
        self.restore_checkpoint = False
        self.mark_success = False
        self.quit_discard = False
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

    # ----- Legacy aliases for backward-compat with older tests/code -----
    @property
    def checkpoint_recording(self) -> bool: return self.take_checkpoint
    @checkpoint_recording.setter
    def checkpoint_recording(self, value: bool) -> None: self.take_checkpoint = value

    @property
    def resume_recording(self) -> bool: return self.restore_checkpoint
    @resume_recording.setter
    def resume_recording(self, value: bool) -> None: self.restore_checkpoint = value

    @property
    def toggle_recording(self) -> bool: return self.mark_success
    @toggle_recording.setter
    def toggle_recording(self, value: bool) -> None: self.mark_success = value

    @property
    def quit_without_saving(self) -> bool: return self.quit_discard
    @quit_without_saving.setter
    def quit_without_saving(self, value: bool) -> None: self.quit_discard = value

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type != self._keyboard_event_type.KEY_PRESS:
            return False
        key_name = event.input.name.upper()
        if key_name == "C":
            self.take_checkpoint = True
            print("[INFO]: Checkpoint requested.")
            return True
        if key_name == "R":
            self.restore_checkpoint = True
            print("[INFO]: Restore-to-checkpoint requested.")
            return True
        if key_name == "Q":
            self.quit_discard = True
            print("[INFO]: Quit-and-discard requested.")
            return True
        if key_name == "S":
            self.mark_success = True
            print("[INFO]: Mark-success-and-save requested.")
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
    """Hold sim commands at a target pose until the real leader arm moves near it.

    Two contexts use this: (1) startup, where the sim is held at the canonical
    init pose so it stays visibly facing the cube until the user moves the
    leader to home; (2) checkpoint resume, where the sim is held at the
    restored pose until the leader is moved back near it.
    """

    def __init__(self, release_threshold: float):
        self.default_threshold = float(release_threshold)
        self.release_threshold = float(release_threshold)
        self._target = None
        self._context = "checkpoint"

    @property
    def active(self) -> bool:
        return self._target is not None

    def activate(
        self,
        target,
        *,
        context: str = "checkpoint",
        release_threshold: float | None = None,
    ) -> None:
        self._target = _clone_value(target)
        self._context = context
        # Allow startup vs. checkpoint to use different tolerances. Falls back
        # to the default when not overridden so a later checkpoint resume does
        # not inherit a wider startup threshold.
        self.release_threshold = (
            float(release_threshold)
            if release_threshold is not None
            else self.default_threshold
        )
        if context == "startup":
            print(
                "[INFO]: Holding sim at home pose (cube-facing). "
                "Move the real leader arm near the home pose to begin live control."
            )
        else:
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
            label = "home" if self._context == "startup" else "checkpoint"
            print(
                f"[INFO]: Leader arm synced to {label} pose; live control "
                f"engaged (max error {error:.4f} rad)."
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
    """Apply pending recording key events and return whether teleop should exit.

    Key semantics:
      * **Q** — quit immediately, discarding the current episode.
      * **S** — mark the current episode a SUCCESS, save it, and exit.
        (Manual counterpart to the auto-detected ``--goal-region`` path.)
      * **C** — checkpoint the current frame so **R** can restore later.
      * **R** — restore the robot pose + env state to the last checkpoint
        AND engage the resume hold at the checkpoint pose. The sim freezes
        at the checkpoint until the operator physically moves the leader
        arm within ``resume_sync_threshold`` of the checkpoint joints;
        only then does live control resume. Without the hold, the leader's
        next read would override the restored pose on frame 1, defeating
        the resume.
    """

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
            print("[WARN]: Restore requested, but --repo-id was not provided.")
        elif checkpoints is not None and checkpoints.has_checkpoint:
            # Snap robot+env to checkpoint, then engage the leader-sync hold
            # at the checkpoint joints. Without the hold, the leader's read
            # on the very next frame would overwrite the restored pose; the
            # hold pins the sim at the checkpoint until the operator moves
            # the real arm into the checkpoint pose.
            hold_target = checkpoints.restore(recorder)
            if hold_target is not None and resume_hold is not None:
                resume_hold.activate(hold_target, context="checkpoint")
        else:
            # Recording always starts at launch in `_cmd_record`, so any
            # R press without a checkpoint is just a no-op the user
            # should be told about.
            print(
                "[WARN]: Restore requested, but no checkpoint has been captured "
                "yet. Press C first to mark a frame, then R to return to it."
            )

    if keyboard.toggle_recording:
        keyboard.toggle_recording = False
        if recorder is None:
            print("[WARN]: Save requested, but --repo-id was not provided.")
        elif recorder.recording:
            recorder.save_episode(success=True)
            print("[SUCCESS]: Manual success — episode saved. Exiting.")
            should_quit = True
        else:
            print("[WARN]: Save requested, but no episode is currently recording.")

    return should_quit


def _success_prompt_response(response: str) -> bool:
    return response.strip().lower() in {"y", "yes"}


def _handle_successful_episode(
    recorder,
    *,
    confirm: bool = True,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Persist a successful episode.

    Defaults to the interactive ``[y/N]`` prompt — the human gets the last
    word on whether a "goal-reached" frame really represents a clean
    demonstration worth keeping. Pass ``confirm=False`` for unattended
    batch-capture (leisaac-style auto-save).
    """
    if recorder is None or not recorder.recording:
        return
    if not confirm:
        recorder.save_episode(success=True)
        print("[SUCCESS]: Goal reached. Episode auto-saved.")
        return
    try:
        response = input_fn("[SUCCESS]: Goal reached. Save this episode? [y/N]: ")
    except EOFError:
        response = ""
    if _success_prompt_response(response):
        recorder.save_episode(success=True)
    else:
        recorder.cancel_episode()


# Backward-compatible alias retained so external callers / tests that
# imported the legacy name keep working.
_prompt_save_successful_episode = _handle_successful_episode


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


def _build_home_target_tensor(device):
    """Return the canonical init pose as an action-ordered tensor (radians).

    The teleop action vector is laid out as
    ``(Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw)`` — see
    :data:`openso101.robots.SO101_SIM_JOINT_NAMES`. The values come from
    :data:`SO101_CANONICAL_INIT_JOINT_POS` in the robot module so any future
    pose tweak there flows through automatically.
    """
    import torch

    from openso101.robots import SO101_SIM_JOINT_NAMES
    from openso101.robots.so101.so_arm101 import SO101_CANONICAL_INIT_JOINT_POS

    values = [SO101_CANONICAL_INIT_JOINT_POS[name] for name in SO101_SIM_JOINT_NAMES]
    return torch.tensor(values, dtype=torch.float32, device=device)


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
    # Default ON so the user sees the explicit place-target during teleop
    # (curriculum sphere chain for pick_place; no-op for tasks without an
    # `object_pose` command term, e.g. stack). Drives both the visualizer
    # and the cube-touches-final-goal auto-success path below.
    args.goal_region = getattr(args, "goal_region", True)
    args.invert_joints = getattr(args, "invert_joints", "")
    args.joint_offsets_deg = getattr(args, "joint_offsets_deg", "")
    args.resume_sync_threshold = getattr(args, "resume_sync_threshold", 0.08)
    # Default OFF: startup hold blocks the sim until the leader physically
    # matches the home pose, which surprises new users (they think teleop is
    # broken). Use --startup-sync to opt in when you specifically want to see
    # the home pose before driving the arm.
    args.startup_sync = getattr(args, "startup_sync", False)
    args.startup_sync_threshold = getattr(args, "startup_sync_threshold", 0.3)
    args.leader_async = getattr(args, "leader_async", True)
    args.auto_save = getattr(args, "auto_save", False)
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
    from openso101.teleop.hdf5_recorder import OpenSO101HDF5TeleopRecorder
    from openso101.teleop.lerobot_interface import LeRobotSO101Leader
    from openso101.teleop.lerobot_recorder import (
        OpenSO101LeRobotRecorder,
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
    leader = None
    startup_error = None
    try:
        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=not args.disable_fabric,
        )
        # Record always runs with teleop action mode + cameras enabled.
        # parse_env_cfg returns the base RL cfg, so apply the variant hooks
        # here before constructing the env. The registry factory only auto-
        # runs the hooks when no explicit cfg is passed; passing cfg=env_cfg
        # means we own the cfg, including its variant state.
        env_cfg.configure_action_mode("teleop")
        env_cfg.configure_cameras(True)

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
        print(
            "[INFO]: Teleop keys: "
            "S=mark SUCCESS + save + exit, "
            "Q=cancel + exit (discard), "
            "C=checkpoint current frame, "
            "R=restore robot+env to checkpoint."
        )

        scene = env.unwrapped.scene
        unwrapped_env = env.unwrapped
        camera_metadata = discover_camera_metadata(scene)
        print(f"[INFO]: Recording cameras: {', '.join(camera_metadata)}")

        checkpoints = None
        if not args.no_record:
            if args.record_format == "hdf5":
                recorder = OpenSO101HDF5TeleopRecorder(
                    root=args.repo_root,
                    task_name=args.task_name,
                    cameras=camera_metadata,
                    fps=args.fps,
                    dataset_id=args.repo_id,
                    sim_joint_names=sim_joint_names,
                    env_id=args.task,
                )
            else:
                recorder = OpenSO101LeRobotRecorder(
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
            print(
                "[INFO]: Recording keys: "
                "S=mark SUCCESS + save + exit, "
                "Q=cancel + exit (discard), "
                "C=checkpoint current frame, "
                "R=restore robot+env to checkpoint."
            )

        leader = LeRobotSO101Leader(
            port=args.leader_port,
            robot_id=args.leader_id,
            inverted_joints=inverted_joints,
            joint_offsets_rad=joint_offsets_rad,
            async_read=bool(getattr(args, "leader_async", True)),
        )
        leader.connect()
        mode = "async (daemon thread)" if leader.async_read else "sync"
        print(
            f"[INFO]: Connected SO101 leader '{args.leader_id}' on "
            f"{args.leader_port} ({mode} read)."
        )

        env.reset()
        if not args.no_camera_viewports:
            open_teleop_viewports(scene)
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

        # Print the canonical home pose in degrees so the user can verify the
        # sim init pose matches what they expect, AND understand what pose the
        # leader needs to reach if --startup-sync is enabled.
        import math as _math
        home_target_rad = _build_home_target_tensor(env.unwrapped.device)
        home_target_deg = [round(_math.degrees(float(v)), 1) for v in home_target_rad.tolist()]
        os.write(
            1,
            (
                f"[INFO]: Sim home pose (deg, action order "
                f"{list(SO101_TELEOP_CONTROL_JOINT_NAMES)}): {home_target_deg}\n"
            ).encode(),
        )

        resume_hold = _TeleopResumeHold(args.resume_sync_threshold)
        if args.startup_sync:
            # Hold the sim at the canonical home pose (cube-facing) until the
            # real leader is moved near it — otherwise the leader's first read
            # overrides env.reset() on frame 1 and the home pose never becomes
            # visible. Threshold defaults wider than the checkpoint-resume
            # threshold because the user has to physically pose the leader.
            resume_hold.activate(
                _build_home_target_tensor(env.unwrapped.device),
                context="startup",
                release_threshold=float(args.startup_sync_threshold),
            )
        target_limiter = _TeleopTargetRateLimiter(args.action_rate_limit)
        control_step = 0
        # Track the worker's read count between profile prints so we can show
        # whether the async daemon is actually polling. In sync mode this
        # stays at 0 and the profile line will report leader_mode=sync.
        prev_async_polls = leader.async_read_count if leader.async_read else 0

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
                    if leader.async_read:
                        cur_polls = leader.async_read_count
                        polls_delta = cur_polls - prev_async_polls
                        prev_async_polls = cur_polls
                        leader_state = (
                            f"leader_mode=async polls_this_interval={polls_delta} "
                            f"total_polls={cur_polls}"
                        )
                    else:
                        leader_state = "leader_mode=sync"
                    print(
                        "[PROFILE]: "
                        f"read={read_ms:.1f}ms step={step_ms:.1f}ms record={record_ms:.1f}ms "
                        f"loop={loop_ms:.1f}ms joint_error_mean={joint_error.mean():.4f}rad "
                        f"joint_error_max={joint_error.max():.4f}rad {leader_state}"
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
                    # Default is the interactive [y/N] prompt; --auto-save
                    # flips to non-interactive auto-save for batch capture.
                    _handle_successful_episode(
                        recorder, confirm=not bool(getattr(args, "auto_save", False))
                    )
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
        if leader is not None:
            # Joins the async-read worker (no-op when async_read=False).
            try:
                leader.close()
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                print(f"[WARN]: Leader cleanup failed: {exc}")
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


# HDF5 export filter defaults. These mirror leisaac's `isaaclab2lerobot.py`
# behavior: the first frames of a teleop episode are dominated by env.reset()
# transients and any startup-hold zero-action lead-in, and very short
# episodes are almost always failed attempts rather than useful demos.
# Filtering both out keeps the IL training signal clean. Set via the
# `--skip-leading-frames` and `--min-episode-frames` CLI flags.
_DEFAULT_SKIP_LEADING_FRAMES = 5
_DEFAULT_MIN_EPISODE_FRAMES = 10


def _push_convert_hdf5_to_lerobot(
    hdf5_root: Path,
    lerobot_root: Path,
    repo_id: str,
    overwrite_export: bool = False,
    skip_leading_frames: int = _DEFAULT_SKIP_LEADING_FRAMES,
    min_episode_frames: int = _DEFAULT_MIN_EPISODE_FRAMES,
    async_flush: bool = True,
) -> Path:
    """Convert HDF5 teleop episodes to a local LeRobot dataset.

    ``async_flush=True`` (default) overlaps the next episode's HDF5 read
    + frame iteration with the previous episode's video encoding. On
    multi-episode pushes this typically halves wall-clock time because
    LeRobot's ``save_episode`` blocks on ffmpeg/torchcodec encode.
    """
    import h5py
    import numpy as np
    import queue
    import threading
    import time

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

    skipped_short: list[str] = []
    exported = 0

    def _flush_episode_sync(name: str) -> None:
        dataset.save_episode()
        print(f"[INFO]: Exported {name} to local LeRobot dataset.")

    # When async_flush is on, we put a sentinel-tagged "save now" job on
    # the queue after the producer finishes adding frames for an
    # episode. The worker drains the queue and calls save_episode() for
    # each tag. Because LeRobotDataset.add_frame() is NOT thread-safe,
    # the producer must wait for the worker to finish the previous save
    # before starting frames for the next episode — i.e., depth-1
    # pipeline (one episode being encoded + one being read). That's
    # still enough to fully overlap read and encode wall-clock.
    worker_thread = None
    flush_queue: "queue.Queue[str | None]" = queue.Queue(maxsize=1)
    worker_error: list[BaseException] = []

    def _flush_worker() -> None:
        while True:
            name = flush_queue.get()
            if name is None:
                flush_queue.task_done()
                return
            try:
                _flush_episode_sync(name)
            except BaseException as exc:  # noqa: BLE001 — relay to main
                worker_error.append(exc)
                flush_queue.task_done()
                return
            flush_queue.task_done()

    if async_flush:
        worker_thread = threading.Thread(
            target=_flush_worker, name="lerobot-flush", daemon=True,
        )
        worker_thread.start()

    def _enqueue_flush(name: str) -> None:
        """Either pipeline the save off-thread (async) or run inline."""
        if worker_thread is None:
            _flush_episode_sync(name)
            return
        # Block briefly if the previous episode is still encoding —
        # depth-1 queue avoids unbounded RAM growth on huge datasets.
        flush_queue.put(name)
        # Surface any worker exception on the producer thread.
        if worker_error:
            raise worker_error[0]

    for episode_file in episode_files:
        with h5py.File(episode_file, "r") as h5:
            task = h5.attrs.get("task", "OpenSO-101 teleoperation")
            frame_count = int(h5["action"].shape[0])
            effective_count = frame_count - skip_leading_frames
            if effective_count < min_episode_frames:
                skipped_short.append(
                    f"{episode_file.name} ({effective_count} usable frames < "
                    f"min {min_episode_frames})"
                )
                continue
            # Wait for any in-flight save to finish before mutating the
            # shared dataset buffer with add_frame() for the next episode.
            if worker_thread is not None:
                flush_queue.join()
                if worker_error:
                    raise worker_error[0]
            for frame_idx in range(skip_leading_frames, frame_count):
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
            _enqueue_flush(episode_file.name)
            exported += 1

    # Drain the worker before we let the function return.
    if worker_thread is not None:
        flush_queue.join()
        flush_queue.put(None)
        worker_thread.join(timeout=60.0)
        if worker_error:
            raise worker_error[0]

    if skipped_short:
        print(
            f"[INFO]: Skipped {len(skipped_short)} short episode(s): "
            + ", ".join(skipped_short)
        )
    if exported == 0:
        raise SystemExit(
            f"No episodes met the minimum length threshold "
            f"(skip_leading={skip_leading_frames}, min_frames={min_episode_frames}). "
            "Lower --min-episode-frames or capture longer demos."
        )
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
            skip_leading_frames=int(getattr(args, "skip_leading_frames",
                                            _DEFAULT_SKIP_LEADING_FRAMES)),
            min_episode_frames=int(getattr(args, "min_episode_frames",
                                           _DEFAULT_MIN_EPISODE_FRAMES)),
            async_flush=not bool(getattr(args, "no_async_flush", False)),
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
        # Recover the gym env ID from the HDF5 attrs written at record time.
        # Without this, replay would silently default to PickPlace even for
        # episodes recorded in Lift or Stack and render the wrong scene.
        import h5py

        with h5py.File(episode_path, "r") as h5:
            env_id = h5.attrs.get("env_id")
        if env_id is None:
            raise SystemExit(
                f"Episode {episode_path} does not record an env_id "
                "(recorded before 2026-05-16 schema bump). Pass "
                "--task explicitly, e.g. --task OpenSO101-Stack-v0."
            )
        args.task = env_id.decode() if isinstance(env_id, bytes) else str(env_id)
        print(f"[INFO]: Replay env auto-selected from episode attrs: {args.task}")

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
        # Replay always runs with teleop action mode + cameras enabled.
        env_cfg.configure_action_mode("teleop")
        env_cfg.configure_cameras(True)

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
# il train  (thin wrapper around `lerobot.scripts.train`)
# ---------------------------------------------------------------------------


def _cmd_train(args: argparse.Namespace) -> int:
    """Train an IL policy via LeRobot's training CLI.

    Thin wrapper that delegates to `openso101.il.runners.train_il_policy`
    so the CLI and the programmatic Python API behave identically.
    """
    from openso101.il.runners import train_il_policy

    result = train_il_policy(
        policy=args.policy,
        dataset=args.dataset,
        output_dir=getattr(args, "output_dir", None) or _default_il_train_output_dir(args.policy),
        repo_id=getattr(args, "repo_id", None),
        steps=getattr(args, "steps", None),
        batch_size=getattr(args, "batch_size", None),
        wandb=bool(getattr(args, "wandb", False)),
        extra_args=getattr(args, "extra_args", None),
    )
    if not result.succeeded:
        print(
            f"[ERROR]: LeRobot trainer exited with code {result.returncode}. "
            "If the error mentions a missing LeRobot install, run "
            "`bash scripts/install.sh` from the repo root."
        )
    return int(result.returncode)


def _default_il_train_output_dir(policy: str) -> str:
    """Mirror RL's logs/ convention for trained IL checkpoints."""
    import time
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    return f"logs/lerobot/openso101_{policy}/{stamp}"


# ---------------------------------------------------------------------------
# il play  (deploy a trained policy in sim)
# ---------------------------------------------------------------------------


def _cmd_play(args: argparse.Namespace) -> int:
    """Roll out a LeRobot checkpoint inside the OpenSO-101 sim env.

    Architecturally identical to ``_cmd_record`` but with the leader
    replaced by ``policy.select_action(observation)``. Cameras are
    forced on because every IL policy in scope (ACT, Diffusion) expects
    visual inputs.
    """
    import os
    import time
    import traceback

    args.no_camera_viewports = getattr(args, "no_camera_viewports", False)
    args.num_envs = getattr(args, "num_envs", None)
    args.steps = getattr(args, "steps", None)

    simulation_app = _launch_isaac_app(args, enable_cameras=True)

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    import openso101.tasks  # noqa: F401
    from openso101.teleop.camera_viewports import open_teleop_viewports

    env = None
    policy = None
    startup_error = None
    try:
        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=not args.disable_fabric,
        )
        # Default to the RL action mode for IL play so episode rewards +
        # terminations stay active — that lets the operator read success
        # signals off the env. Pass --action-mode teleop only when you
        # want the long single-episode behavior teleop uses.
        env_cfg.configure_action_mode(getattr(args, "action_mode", "rl"))
        env_cfg.configure_cameras(True)
        if args.num_envs is None and hasattr(env_cfg, "scene"):
            env_cfg.scene.num_envs = 1

        env = gym.make(args.task, cfg=env_cfg)
        scene = env.unwrapped.scene
        unwrapped_env = env.unwrapped

        policy = _load_lerobot_policy(args.policy_path, device=env.unwrapped.device)
        print(f"[INFO]: Loaded LeRobot policy from {args.policy_path}.")
        if hasattr(policy, "reset"):
            policy.reset()

        env.reset()
        if not args.no_camera_viewports:
            open_teleop_viewports(scene)

        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        step = 0
        max_steps = int(args.steps) if args.steps is not None else None
        while simulation_app.is_running():
            with torch.inference_mode():
                obs = _build_il_policy_observation(unwrapped_env, scene)
                action = policy.select_action(obs)
                # Policy returns shape (1, action_dim) on most LeRobot
                # checkpoints. Squeeze to the action-space shape that
                # ``env.step`` expects.
                action = action.to(env.unwrapped.device).reshape(actions.shape)
                actions.copy_(action)
                env.step(actions)
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        print(f"[INFO]: IL play loop exited after {step} steps.")
    except Exception as exc:
        startup_error = exc
        print("[ERROR]: IL play failed before clean shutdown:", flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    finally:
        if env is not None:
            env.close()
        if startup_error is None:
            simulation_app.close()

    if startup_error is not None:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    return 0


def _load_lerobot_policy(checkpoint_path: str, *, device):
    """Load a LeRobot policy checkpoint via the shared `il.policies` API.

    Both `_cmd_play` (sim) and `openso101 sim2real deploy` (real) go
    through `openso101.il.policies.load_policy` so behaviour stays
    identical between the two contexts.
    """
    try:
        from openso101.il.policies import load_policy
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is required to load IL policies. Install it via "
            "`bash scripts/install.sh` or `pip install \"lerobot[feetech]==0.4.0\"`."
        ) from exc

    device_str = str(device) if device is not None else None
    return load_policy(checkpoint_path, device=device_str)


def _build_il_policy_observation(unwrapped_env, scene) -> dict:
    """Translate the sim's env state into LeRobot's expected obs schema.

    LeRobot policies trained on the dataset our `il push` produces want:
      * ``observation.state`` — joint positions in motor units.
      * ``observation.images.<camera>`` — uint8 RGB (H, W, 3) per camera.

    The camera names must match what the dataset was recorded with —
    the same ``wrist_camera`` / ``overhead_camera`` keys used by
    :class:`OpenSO101HDF5TeleopRecorder`.
    """
    import numpy as np
    import torch

    from openso101.teleop.lerobot_recorder import collect_camera_buffers, read_robot_proprio
    from openso101.teleop.so101_mapping import (
        batched_action_to_motor_units,
        get_sim_joint_names,
    )

    sim_joint_names = get_sim_joint_names()
    qpos_rad, _ = read_robot_proprio(scene["robot"], sim_joint_names=sim_joint_names)
    qpos_motor = batched_action_to_motor_units(
        torch.as_tensor(qpos_rad, dtype=torch.float32)
    ).numpy().astype(np.float32, copy=False)

    obs: dict = {
        "observation.state": torch.from_numpy(qpos_motor).unsqueeze(0),
    }
    for cam_name, frame in collect_camera_buffers(scene).items():
        # LeRobot expects images as float tensors in [0, 1] with shape
        # (1, 3, H, W) by default; the policy's normalize step handles
        # the actual standardization. We just match shape + dtype.
        arr = np.asarray(frame, dtype=np.uint8)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        obs[f"observation.images.{cam_name}"] = tensor
    return obs


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
    # Startup home-pose hold: keeps sim at the cube-facing init pose until the
    # real leader is moved near home. Off by default — if the user's leader
    # isn't near our home pose at launch, the hold blocks teleop entirely and
    # looks like a dead-leader bug. Enable only when you specifically want the
    # home pose to be visible before live control engages.
    p_rec.add_argument(
        "--startup-sync",
        dest="startup_sync",
        action="store_true",
        default=False,
        help="Hold sim at home pose until the real leader is moved near it.",
    )
    p_rec.add_argument(
        "--startup-sync-threshold",
        type=float,
        default=0.3,
        help="Max per-joint radian error to release startup hold (default 0.3 rad ≈ 17°).",
    )
    p_rec.add_argument(
        "--auto-save",
        action="store_true",
        default=False,
        help=(
            "Skip the [y/N] prompt on goal success and persist every "
            "auto-detected successful episode. Default is the interactive "
            "prompt — pass this only for unattended batch capture sessions."
        ),
    )
    # Async leader read: a daemon thread continuously polls the Feetech bus
    # and caches the latest reading, so the sim step doesn't block on serial
    # I/O. Restores the latency feature lost during the safe_sim2real port.
    p_rec.add_argument(
        "--no-leader-async",
        dest="leader_async",
        action="store_false",
        default=True,
        help=(
            "Disable the leader-read worker thread; fall back to a "
            "synchronous Feetech bus read on every sim step. Use if the "
            "worker thread misbehaves on your hardware."
        ),
    )
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
    p_push.add_argument(
        "--skip-leading-frames",
        type=int,
        default=_DEFAULT_SKIP_LEADING_FRAMES,
        help=(
            "Drop the first N frames of each episode before LeRobot export. "
            "These are dominated by env.reset() transients and startup-hold "
            "zero-action frames that pollute IL training signal. "
            f"Default: {_DEFAULT_SKIP_LEADING_FRAMES}. Use 0 to keep all frames."
        ),
    )
    p_push.add_argument(
        "--min-episode-frames",
        type=int,
        default=_DEFAULT_MIN_EPISODE_FRAMES,
        help=(
            "Drop episodes whose post-skip frame count is below this threshold. "
            "Short episodes are almost always failed attempts. "
            f"Default: {_DEFAULT_MIN_EPISODE_FRAMES}."
        ),
    )
    p_push.add_argument(
        "--no-async-flush",
        action="store_true",
        default=False,
        help=(
            "Disable the per-episode video-encode worker thread (default ON). "
            "With async flush, the next episode's HDF5 read overlaps the "
            "previous episode's LeRobot save_episode() encode. Useful to "
            "disable when debugging or on memory-constrained machines."
        ),
    )
    p_push.set_defaults(func=_cmd_push)

    p_train = sub.add_parser(
        "train",
        help="Train an IL policy via LeRobot's trainer (ACT or Diffusion).",
    )
    p_train.add_argument(
        "--policy",
        required=True,
        choices=["act", "diffusion"],
        help="LeRobot policy class to train. ACT (action chunking transformer) "
             "and Diffusion are both supported.",
    )
    p_train.add_argument(
        "--dataset",
        required=True,
        help="LeRobot dataset: either a Hub repo_id (e.g. 'user/openso101_pick') "
             "or a local directory produced by 'openso101 il push'.",
    )
    p_train.add_argument(
        "--repo-id",
        default=None,
        help="Override the repo_id passed to LeRobot when --dataset is a local path. "
             "Defaults to 'local/<dataset-dir-name>'.",
    )
    p_train.add_argument(
        "--output-dir",
        default=None,
        help="Where to write checkpoints + logs. Defaults to "
             "logs/lerobot/openso101_<policy>/<timestamp>/.",
    )
    p_train.add_argument("--steps", type=int, default=None, help="Override LeRobot's default training steps.")
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p_train.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed verbatim to `lerobot.scripts.train` after `--`.",
    )
    p_train.set_defaults(func=_cmd_train)

    p_play = sub.add_parser(
        "play",
        help="Roll out a trained IL policy in the OpenSO-101 sim env.",
    )
    p_play.add_argument("--task", required=True)
    p_play.add_argument(
        "--policy-path",
        required=True,
        help="Path to the LeRobot pretrained-model dir (or its parent output_dir).",
    )
    p_play.add_argument("--num-envs", type=int, default=None)
    p_play.add_argument("--steps", type=int, default=None, help="Max sim steps; defaults to run until window closed.")
    p_play.add_argument("--no-camera-viewports", action="store_true")
    p_play.add_argument(
        "--action-mode",
        default="rl",
        choices=("rl", "teleop"),
        help=(
            "Env variant. 'rl' (default) keeps rewards + terminations so the "
            "operator can read success signals from the env; 'teleop' uses the "
            "long-episode no-rewards variant matching `il record`."
        ),
    )
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
