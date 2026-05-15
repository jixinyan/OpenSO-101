# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""rsl_rl CLI argument helpers.

SKELETON: bodies not yet ported. Source reference:
/data/safe_sim2real/src/safe_sim2real/scripts/rsl_rl/cli_args.py
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard rsl_rl flag group (--experiment_name, --resume, --load_run,
    --checkpoint, --logger, --log_project_name, ...) to ``parser``."""
    raise NotImplementedError(
        "add_rsl_rl_args body not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/scripts/rsl_rl/cli_args.py"
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> "RslRlBaseRunnerCfg":
    """Resolve the rsl_rl runner cfg for ``task_name`` and apply CLI overrides."""
    raise NotImplementedError(
        "parse_rsl_rl_cfg body not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/scripts/rsl_rl/cli_args.py"
    )


def update_rsl_rl_cfg(
    agent_cfg: "RslRlBaseRunnerCfg",
    args_cli: argparse.Namespace,
) -> "RslRlBaseRunnerCfg":
    """Apply CLI overrides (seed, resume, load_run, checkpoint, run_name, logger,
    log_project_name) onto ``agent_cfg`` in place and return it."""
    raise NotImplementedError(
        "update_rsl_rl_cfg body not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/scripts/rsl_rl/cli_args.py"
    )


__all__ = ["add_rsl_rl_args", "parse_rsl_rl_cfg", "update_rsl_rl_cfg"]
