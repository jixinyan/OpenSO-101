# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""`openso101 data ...` subcommands — all deferred to sub-project F."""

from __future__ import annotations

import argparse


_NOT_IMPL = (
    "openso101 data {cmd}: not implemented in this refactor. "
    "See sub-project F (docs/superpowers/specs/2026-05-13-openso101-refactor-design.md § 13)."
)


def _cmd_generate(args: argparse.Namespace) -> int:
    print(_NOT_IMPL.format(cmd="generate"))
    return 2


def _cmd_inspect(args: argparse.Namespace) -> int:
    print(_NOT_IMPL.format(cmd="inspect"))
    return 2


def add_subparsers(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="data_cmd", required=True)

    p_gen = sub.add_parser("generate", help="Generate synthetic demos")
    p_gen.add_argument("--source", required=True)
    p_gen.add_argument(
        "--backend", required=True, choices=["mimicgen", "isaaclab_mimic"]
    )
    p_gen.add_argument("--output", required=True)
    p_gen.set_defaults(func=_cmd_generate)

    p_ins = sub.add_parser("inspect", help="Inspect a dataset")
    p_ins.add_argument("--dataset", required=True)
    p_ins.set_defaults(func=_cmd_inspect)
