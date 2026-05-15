# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""`openso101 envs ...` subcommands."""

from __future__ import annotations

import argparse


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


def _cmd_random(args: argparse.Namespace) -> int:
    import gymnasium as gym

    app = _launch_isaac_app(headless=False)
    try:
        env = gym.make(args.task, cameras=args.with_cameras)
        env.reset(seed=0)
        for _ in range(args.steps):
            env.step(env.action_space.sample())
        env.close()
    finally:
        if app is not None:
            app.close()
    return 0


def _cmd_zero(args: argparse.Namespace) -> int:
    import gymnasium as gym
    import numpy as np

    app = _launch_isaac_app(headless=False)
    try:
        env = gym.make(args.task, cameras=args.with_cameras)
        env.reset(seed=0)
        zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        for _ in range(args.steps):
            env.step(zero_action)
        env.close()
    finally:
        if app is not None:
            app.close()
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    import gymnasium as gym

    app = _launch_isaac_app(headless=False)
    try:
        env = gym.make(args.task, cameras=True)
        env.reset(seed=0)
        for _ in range(args.steps):
            env.step(env.action_space.sample())
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
    p_random.set_defaults(func=_cmd_random)

    p_zero = sub.add_parser("zero", help="Run N zero-action steps")
    p_zero.add_argument("--task", required=True)
    p_zero.add_argument("--with-cameras", action="store_true")
    p_zero.add_argument("--steps", type=int, default=100)
    p_zero.set_defaults(func=_cmd_zero)

    p_prev = sub.add_parser("preview", help="Open env with cameras, render frames")
    p_prev.add_argument("--task", required=True)
    p_prev.add_argument("--steps", type=int, default=30)
    p_prev.add_argument("--overhead_pos", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None)
    p_prev.add_argument("--overhead_rpy", type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW"), default=None)
    p_prev.add_argument("--wrist_pos", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None)
    p_prev.add_argument("--wrist_rpy", type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW"), default=None)
    p_prev.set_defaults(func=_cmd_preview)
