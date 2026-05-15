# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""`openso101 envs ...` subcommands."""

from __future__ import annotations

import argparse
import sys


_PREFIX = "OpenSO101-"


def _launch_isaac_app(headless: bool = True):
    """Launch Isaac Sim's SimulationApp and import OpenSO-101 tasks.

    Returns the `SimulationApp` handle (caller is responsible for `.close()`).
    Returns None if `isaaclab` is unavailable (skeleton-only test envs).
    """
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        return None
    launcher = AppLauncher(headless=headless)
    # Trigger gym.register calls for the built-in tasks.
    import openso101.tasks  # noqa: F401
    return launcher.app


def list_envs() -> list[str]:
    """Return registered OpenSO-101 task IDs, sorted."""
    import gymnasium as gym

    return sorted(eid for eid in gym.envs.registry if eid.startswith(_PREFIX))


def _cmd_list(args: argparse.Namespace) -> int:
    import sys
    app = _launch_isaac_app(headless=True)
    try:
        # Isaac Sim hijacks stdout buffering, so write directly to fd 1.
        out = sys.__stdout__ or sys.stdout
        for eid in list_envs():
            out.write(eid + "\n")
        out.flush()
    finally:
        if app is not None:
            app.close()
    return 0


def _build_smoke_env(args, *, force_cameras: bool = False):
    """Construct a small play-mode env for smoke commands.

    Defaults to num_envs=1 (overrideable via --num-envs) so GUI startup is
    fast and 8 GB-class GPUs don't OOM. Always uses play=True (no DR).
    """
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    num_envs = max(1, int(getattr(args, "num_envs", 1) or 1))
    cameras = bool(force_cameras or getattr(args, "with_cameras", False))

    # parse_env_cfg returns the base RL cfg; apply variant hooks here so
    # the smoke runs in play mode (no DR) and at the requested env count.
    env_cfg = parse_env_cfg(args.task, num_envs=num_envs)
    env_cfg.configure_play(True)
    if cameras:
        env_cfg.configure_cameras(True)
    # configure_play set num_envs to its play default (50); honor the
    # user's --num-envs by re-applying.
    env_cfg.scene.num_envs = num_envs

    return gym.make(args.task, cfg=env_cfg)


def _step_loop(env, action_fn, steps: int) -> None:
    """Step `env` for `steps` iterations using `action_fn(env) -> action`,
    heartbeating progress to fd 1 (Isaac Sim's stdio hijack buffers BOTH
    sys.stdout and sys.__stdout__ after sim boot — but a direct os.write
    to file descriptor 1 still reaches the terminal)."""
    import os
    def emit(msg: str) -> None:
        os.write(1, (msg + "\n").encode())
    emit(f"[envs] Stepping {steps} actions...")
    env.reset(seed=0)
    emit("[envs] env.reset() complete; entering step loop")
    for i in range(steps):
        env.step(action_fn(env))
        if (i + 1) % max(1, steps // 5) == 0 or i == 0:
            emit(f"[envs] step {i + 1}/{steps}")
    emit("[envs] Done.")


def _cmd_random(args: argparse.Namespace) -> int:
    import sys
    app = _launch_isaac_app(headless=False)
    try:
        env = _build_smoke_env(args)
        _step_loop(env, lambda e: e.action_space.sample(), args.steps)
        env.close()
    finally:
        if app is not None:
            app.close()
    return 0


def _cmd_zero(args: argparse.Namespace) -> int:
    import numpy as np
    import sys
    app = _launch_isaac_app(headless=False)
    try:
        env = _build_smoke_env(args)
        zero = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        _step_loop(env, lambda e: zero, args.steps)
        env.close()
    finally:
        if app is not None:
            app.close()
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    import sys
    app = _launch_isaac_app(headless=False)
    try:
        env = _build_smoke_env(args, force_cameras=True)
        _step_loop(env, lambda e: e.action_space.sample(), args.steps)
        env.close()
    finally:
        if app is not None:
            app.close()
    return 0


def add_subparsers(parser: argparse.ArgumentParser) -> None:
    """Register `envs <sub>` parsers on the given parent parser."""
    sub = parser.add_subparsers(dest="envs_cmd", required=True)

    p_list = sub.add_parser("list", help="List registered OpenSO-101 tasks")
    p_list.set_defaults(func=_cmd_list)

    p_random = sub.add_parser("random", help="Run N random-action steps")
    p_random.add_argument("--task", required=True)
    p_random.add_argument("--with-cameras", action="store_true")
    p_random.add_argument("--steps", type=int, default=100)
    p_random.add_argument("--num-envs", type=int, default=1)
    p_random.set_defaults(func=_cmd_random)

    p_zero = sub.add_parser("zero", help="Run N zero-action steps")
    p_zero.add_argument("--task", required=True)
    p_zero.add_argument("--with-cameras", action="store_true")
    p_zero.add_argument("--steps", type=int, default=100)
    p_zero.add_argument("--num-envs", type=int, default=1)
    p_zero.set_defaults(func=_cmd_zero)

    p_prev = sub.add_parser("preview", help="Open env with cameras, render frames")
    p_prev.add_argument("--task", required=True)
    p_prev.add_argument("--steps", type=int, default=30)
    p_prev.add_argument("--num-envs", type=int, default=1)
    p_prev.add_argument("--overhead_pos", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None)
    p_prev.add_argument("--overhead_rpy", type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW"), default=None)
    p_prev.add_argument("--wrist_pos", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None)
    p_prev.add_argument("--wrist_rpy", type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW"), default=None)
    p_prev.set_defaults(func=_cmd_preview)
