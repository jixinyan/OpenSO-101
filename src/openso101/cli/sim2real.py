# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""`openso101 sim2real ...` subcommands.

The single command — ``deploy`` — loads a LeRobot policy checkpoint and
runs it on the real SO-101 follower arm. The same checkpoint format
produced by ``openso101 il train`` and consumed by ``openso101 il play``
in sim is consumed here on real hardware, so a trained policy
transfers without re-export.
"""

from __future__ import annotations

import argparse


def _cmd_deploy(args: argparse.Namespace) -> int:
    """Dispatch to the deploy implementation.

    Implementation lives in :mod:`openso101.sim2real.deploy` so the
    heavy LeRobot + cv2 imports don't fire on `--help`.
    """
    from openso101.sim2real.deploy import deploy

    return deploy(args)


def add_subparsers(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="sim2real_cmd", required=True)

    p_dep = sub.add_parser(
        "deploy",
        help="Roll out a trained IL policy on the real SO-101 follower.",
    )
    p_dep.add_argument(
        "--policy-path",
        required=True,
        help="Path to the LeRobot pretrained-model dir (or its parent output_dir, "
             "same convention as `il play --policy-path`).",
    )
    p_dep.add_argument(
        "--follower-port",
        required=True,
        help="Serial port of the SO-101 follower (e.g. /dev/ttyACM1).",
    )
    p_dep.add_argument(
        "--follower-id",
        required=True,
        help="LeRobot follower-arm ID (matches the calibration JSON name).",
    )
    p_dep.add_argument(
        "--wrist-camera-index",
        type=int,
        default=0,
        help="OpenCV index of the wrist camera (cv2.VideoCapture(index)).",
    )
    p_dep.add_argument(
        "--overhead-camera-index",
        type=int,
        default=2,
        help="OpenCV index of the overhead camera.",
    )
    p_dep.add_argument(
        "--camera-width",
        type=int,
        default=128,
        help="Width to which camera frames are resized before inference. "
             "Must match what the dataset was recorded with.",
    )
    p_dep.add_argument(
        "--camera-height",
        type=int,
        default=128,
        help="Height to which camera frames are resized before inference.",
    )
    p_dep.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target control rate. Should match the dataset fps so the "
             "policy sees its training temporal distribution.",
    )
    p_dep.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Exit after N inference steps. Defaults to run forever "
             "until Ctrl+C.",
    )
    p_dep.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for policy inference. Use 'cpu' if you don't "
             "have a GPU on the deploy host.",
    )
    p_dep.add_argument(
        "--profile",
        action="store_true",
        help="Print effective control-loop rate every --profile-interval steps.",
    )
    p_dep.add_argument(
        "--profile-interval",
        type=int,
        default=30,
        help="Steps between profile prints.",
    )
    p_dep.set_defaults(func=_cmd_deploy)
