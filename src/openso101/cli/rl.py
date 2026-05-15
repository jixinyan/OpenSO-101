# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""`openso101 rl ...` subcommands.

train --algo selects the RL algorithm:
- ppo: real (currently a skeleton stub pending source port).
- sac: deferred to sub-project E.
- ppo_lag / cpo / focops: NOT supported (safe-RL stays in safe_sim2real).
"""

from __future__ import annotations

import argparse


_SAFE_ALGOS = ("ppo_lag", "cpo", "focops")
_OPENSO101_ALGOS = ("ppo", "sac")
_ALL_ALGOS = _OPENSO101_ALGOS + _SAFE_ALGOS


def _cmd_train(args: argparse.Namespace) -> int:
    if args.algo in _SAFE_ALGOS:
        print(
            f"openso101 rl train: --algo {args.algo} is not available in "
            "OpenSO-101. Safe-RL stays in the legacy safe_sim2real repository."
        )
        return 2

    if args.algo == "sac":
        print(
            "openso101 rl train: --algo sac is part of sub-project E "
            "(off-policy RL) and not yet implemented in this refactor."
        )
        return 2

    if args.algo == "ppo":
        # SKELETON: real PPO training body ports from
        # /data/safe_sim2real/src/safe_sim2real/scripts/rsl_rl/train.py
        # once the legacy source stabilizes.
        raise NotImplementedError(
            "openso101 rl train --algo ppo: training body not yet ported. "
            "Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/scripts/rsl_rl/train.py"
        )

    # Should be unreachable due to argparse choices.
    print(f"openso101 rl train: unknown algorithm {args.algo!r}")
    return 2


def _cmd_play(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "openso101 rl play: not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/scripts/rsl_rl/play.py"
    )


def _cmd_plot(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "openso101 rl plot: not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/scripts/plot_training.py"
    )


def add_subparsers(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="rl_cmd", required=True)

    p_train = sub.add_parser("train", help="Train an RL policy")
    p_train.add_argument("--task", required=True, help="Gym ID")
    p_train.add_argument(
        "--algo",
        required=True,
        choices=_ALL_ALGOS,
        help="Algorithm (ppo|sac|ppo_lag|cpo|focops)",
    )
    p_train.add_argument("--with-cameras", action="store_true")
    p_train.add_argument("--headless", action="store_true")
    p_train.set_defaults(func=_cmd_train)

    p_play = sub.add_parser("play", help="Replay a trained checkpoint")
    p_play.add_argument("--task", required=True)
    p_play.add_argument("--checkpoint", required=True)
    p_play.add_argument("--with-cameras", action="store_true")
    p_play.set_defaults(func=_cmd_play)

    p_plot = sub.add_parser("plot", help="Plot training curves from a run dir")
    p_plot.add_argument("--run", required=True)
    p_plot.set_defaults(func=_cmd_plot)
