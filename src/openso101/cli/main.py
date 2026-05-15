# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""`openso101` top-level CLI entry point."""

from __future__ import annotations

import argparse
import sys


def _not_implemented_yet(group_name: str, sub_project: str):
    """Factory: returns an argparse handler that prints a deferral message."""

    def _handler(args: argparse.Namespace) -> int:
        print(
            f"openso101 {group_name}: command group not yet implemented "
            f"in this CLI build (see sub-project {sub_project} in "
            f"docs/superpowers/specs/2026-05-13-openso101-refactor-design.md § 13).",
        )
        return 2

    return _handler


def build_parser() -> argparse.ArgumentParser:
    """Construct the `openso101` argparse tree."""
    parser = argparse.ArgumentParser(
        prog="openso101",
        description="OpenSO-101: open-source robot learning framework for the SO-101.",
    )
    sub = parser.add_subparsers(dest="group", required=True)

    # envs group — fully implemented in this task.
    # Import here to avoid forcing gymnasium-on-import for `--help`.
    from . import envs as envs_cli

    # Trigger built-in task registration on CLI startup so `envs list` works.
    import openso101  # noqa: F401

    p_envs = sub.add_parser("envs", help="Task discovery and sanity checks")
    envs_cli.add_subparsers(p_envs)

    # rl group — fully implemented in Task 16.
    from . import rl as rl_cli

    p_rl = sub.add_parser("rl", help="RL training, playback, plotting")
    rl_cli.add_subparsers(p_rl)

    # Placeholders for il/data/sim2real groups (wired up in Task 17+).
    for group, sub_project in (
        ("il", "C"),
        ("data", "F"),
        ("sim2real", "future"),
    ):
        p = sub.add_parser(group, help=f"{group} subcommands (not yet wired)")
        p.set_defaults(func=_not_implemented_yet(group, sub_project))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
