# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""`openso101 il ...` subcommands."""

from __future__ import annotations

import argparse


def _cmd_record(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "openso101 il record: not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/scripts/lerobot/teleop_agent.py"
    )


def _cmd_push(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "openso101 il push: not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/scripts/lerobot/push_dataset.py"
    )


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


def _cmd_replay(args: argparse.Namespace) -> int:
    print(
        "openso101 il replay: not implemented in this refactor. "
        "See sub-project C."
    )
    return 2


def add_subparsers(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="il_cmd", required=True)

    p_rec = sub.add_parser("record", help="Record teleop demonstrations")
    p_rec.add_argument("--task", required=True)
    p_rec.add_argument("--leader-port", required=True)
    p_rec.add_argument("--leader-id", required=True)
    p_rec.add_argument("--repo-id")
    p_rec.add_argument("--repo-root")
    p_rec.add_argument("--task-name")
    p_rec.add_argument("--profile-teleop", action="store_true")
    p_rec.add_argument("--no-camera-viewports", action="store_true")
    p_rec.set_defaults(func=_cmd_record)

    p_push = sub.add_parser("push", help="Push HDF5 dataset to LeRobot Hub")
    p_push.add_argument("--repo-root", required=True)
    p_push.add_argument("--repo-id", required=True)
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
    p_replay.set_defaults(func=_cmd_replay)
