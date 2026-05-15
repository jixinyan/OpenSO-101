# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""`openso101 sim2real ...` subcommands — placeholder for real-arm deployment."""

from __future__ import annotations

import argparse


def _cmd_deploy(args: argparse.Namespace) -> int:
    print(
        "openso101 sim2real deploy: real-arm deployment is a future "
        "sub-project. See docs/superpowers/specs/2026-05-13-openso101-refactor-design.md § 13."
    )
    return 2


def add_subparsers(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="sim2real_cmd", required=True)

    p_dep = sub.add_parser("deploy", help="Deploy a policy to the real arm")
    p_dep.add_argument("--policy-path", required=True)
    p_dep.set_defaults(func=_cmd_deploy)
