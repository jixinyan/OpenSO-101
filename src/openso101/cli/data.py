# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""`openso101 data ...` subcommands — all deferred to sub-project F."""

from __future__ import annotations

import argparse


# Synthetic data generation is intentionally deferred until the RL + IL
# pipelines are fully debugged. The CLI surface here mirrors the
# documented intent in docs/concepts/data_generation.md so the eventual
# implementation lands behind a stable argparse contract.
_NOT_IMPL = (
    "openso101 data {cmd}: synthetic data generation is deferred until the "
    "RL + IL pipelines are fully validated. The CLI surface is stable; the "
    "generator bodies are planned in openso101/data_gen/{{mimicgen,"
    "isaaclab_mimic}}/generator.py."
)


def _cmd_generate(args: argparse.Namespace) -> int:
    print(_NOT_IMPL.format(cmd="generate"))
    return 2


def _cmd_inspect(args: argparse.Namespace) -> int:
    print(_NOT_IMPL.format(cmd="inspect"))
    return 2


def add_subparsers(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="data_cmd", required=True)

    p_gen = sub.add_parser(
        "generate",
        help="Generate synthetic demos by perturbing a seed dataset.",
    )
    p_gen.add_argument(
        "--task",
        required=True,
        help="Gym task ID to spawn for synthetic rollouts (e.g. OpenSO101-PickPlace-v0).",
    )
    p_gen.add_argument(
        "--backend",
        required=True,
        choices=["mimicgen", "isaaclab_mimic"],
        help="Synthetic generation backend. mimicgen is the mature path; "
             "isaaclab_mimic is the first-party Isaac Lab integration.",
    )
    p_gen.add_argument(
        "--seed-dataset",
        required=True,
        help="LeRobot seed dataset (Hub repo_id OR local directory path).",
    )
    p_gen.add_argument(
        "--num-trials",
        type=int,
        default=1000,
        help="How many perturbed rollouts to attempt; only successful ones are kept.",
    )
    p_gen.add_argument(
        "--output-dir",
        required=True,
        help="Where to write the augmented LeRobot dataset.",
    )
    p_gen.set_defaults(func=_cmd_generate)

    p_ins = sub.add_parser(
        "inspect",
        help="Print summary stats for a generated dataset.",
    )
    p_ins.add_argument(
        "--dataset",
        required=True,
        help="LeRobot dataset to inspect (Hub repo_id OR local directory).",
    )
    p_ins.set_defaults(func=_cmd_inspect)
