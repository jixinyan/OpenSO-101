# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

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

    p_envs = sub.add_parser("envs", help="Task discovery and sanity checks")
    envs_cli.add_subparsers(p_envs)

    # rl group — fully implemented in Task 16.
    from . import rl as rl_cli

    p_rl = sub.add_parser("rl", help="RL training, playback, plotting")
    rl_cli.add_subparsers(p_rl)

    # il + sim2real groups.
    from . import il as il_cli
    from . import sim2real as sim2real_cli

    p_il = sub.add_parser("il", help="Teleop, datasets, IL training")
    il_cli.add_subparsers(p_il)

    p_sim2real = sub.add_parser("sim2real", help="Sim-to-real deployment on a real SO-101")
    sim2real_cli.add_subparsers(p_sim2real)

    from . import sim2sim as sim2sim_cli

    p_sim2sim = sub.add_parser("sim2sim", help="MuJoCo 策略与物理评估")
    sim2sim_cli.add_subparsers(p_sim2sim)

    from . import scenes as scenes_cli

    p_scenes = sub.add_parser("scenes", help="Objaverse assets and custom scenes")
    scenes_cli.add_subparsers(p_scenes)

    from . import validate as validate_cli

    p_validate = sub.add_parser("validate", help="源码检查与统一运行验证")
    validate_cli.add_subparsers(p_validate)

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
