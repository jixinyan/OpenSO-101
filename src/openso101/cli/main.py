# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""`openso101` top-level CLI entry point."""

from __future__ import annotations

import argparse
import sys



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

    # il/data/sim2real groups — wired up in Task 17.
    from . import il as il_cli
    from . import data as data_cli
    from . import sim2real as sim2real_cli

    p_il = sub.add_parser("il", help="Teleop, datasets, IL training")
    il_cli.add_subparsers(p_il)

    p_data = sub.add_parser("data", help="Synthetic data generation (sub-project F)")
    data_cli.add_subparsers(p_data)

    p_sim2real = sub.add_parser("sim2real", help="Sim-to-real / deployment (future)")
    sim2real_cli.add_subparsers(p_sim2real)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Hydra suppresses full tracebacks by default and tells you to set this
    # env var — preempt that prompt so users see the real traceback first time.
    import os
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
