# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Tests for the openso101 rl CLI surface (dispatch + rejection messages).

Subprocess-based to verify the wired-up entry point end-to-end.
"""

from __future__ import annotations

import subprocess
import sys


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "openso101.cli.main", *args],
        capture_output=True,
        text=True,
        cwd="/data/OpenSO-101",
        env={"PYTHONPATH": "/data/OpenSO-101/src", "PATH": "/usr/bin:/bin"},
    )


def test_safe_rl_algos_are_not_choices():
    """Safe-RL algorithms (PPO-Lag, CPO, FOCOPS) are not part of OpenSO-101."""
    for algo in ("ppo_lag", "cpo", "focops"):
        proc = _run_cli(
            "rl", "train",
            "--task", "OpenSO101-PickPlace-v0",
            "--algo", algo,
            "--headless",
        )
        combined = proc.stdout + proc.stderr
        # argparse rejects unknown choices before our dispatch runs.
        assert proc.returncode != 0, (algo, combined)
        assert "invalid choice" in combined.lower(), (algo, combined)


def test_sac_is_not_a_choice():
    """OpenSO-101 ships PPO only — SAC is no longer advertised as a future."""
    proc = _run_cli(
        "rl", "train",
        "--task", "OpenSO101-PickPlace-v0",
        "--algo", "sac",
        "--headless",
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "invalid choice" in combined.lower(), combined


def test_unknown_algo_is_argparse_error():
    # argparse choices=[...] should reject unknown values before our dispatch runs.
    proc = _run_cli(
        "rl", "train",
        "--task", "OpenSO101-PickPlace-v0",
        "--algo", "totally_not_an_algorithm",
    )
    # argparse usage errors exit with 2 and print to stderr.
    assert proc.returncode != 0
    assert "invalid choice" in (proc.stdout + proc.stderr).lower()


def test_distillation_is_a_choice():
    """`--algo distillation` must parse — actual dispatch needs a teacher."""
    proc = _run_cli(
        "rl", "train",
        "--task", "OpenSO101-PickPlace-v0",
        "--algo", "distillation",
        "--headless",
    )
    combined = proc.stdout + proc.stderr
    # argparse must accept the choice; dispatch then rejects (no teacher).
    assert "invalid choice" not in combined.lower(), combined
    assert proc.returncode == 2, combined
    assert "teacher-checkpoint" in combined.lower(), combined


def test_distillation_requires_existing_teacher_checkpoint(tmp_path):
    """A non-existent teacher path must surface a clear FileNotFoundError."""
    proc = _run_cli(
        "rl", "train",
        "--task", "OpenSO101-PickPlace-v0",
        "--algo", "distillation",
        "--teacher-checkpoint", str(tmp_path / "no-such-teacher.pt"),
        "--headless",
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    # Either we hit the explicit FileNotFoundError, or the isaaclab import
    # fails first in CI environments without Isaac Sim. Both prove
    # dispatch fired past argparse.
    assert (
        "no-such-teacher" in combined
        or "FileNotFoundError" in combined
        or "isaaclab" in combined.lower()
    ), combined


def test_distillation_algo_routes_to_dedicated_entry_point():
    """Sanity-check the algo-to-entry-point map exposed by the module."""
    from openso101.cli import rl as rl_cli

    assert rl_cli._ALGO_TO_ENTRY_POINT["ppo"] == "rsl_rl_cfg_entry_point"
    assert (
        rl_cli._ALGO_TO_ENTRY_POINT["distillation"]
        == "rsl_rl_distillation_cfg_entry_point"
    )
