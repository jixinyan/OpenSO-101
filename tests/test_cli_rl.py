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


def test_safe_rl_algos_rejected_with_clear_message():
    for algo in ("ppo_lag", "cpo", "focops"):
        proc = _run_cli(
            "rl", "train",
            "--task", "OpenSO101-PickPlace-v0",
            "--algo", algo,
            "--headless",
        )
        combined = proc.stdout + proc.stderr
        assert proc.returncode == 2, (algo, proc.returncode, combined)
        assert "not available in OpenSO-101" in combined, combined
        assert "safe_sim2real" in combined, combined


def test_sac_says_sub_project_e():
    proc = _run_cli(
        "rl", "train",
        "--task", "OpenSO101-PickPlace-v0",
        "--algo", "sac",
        "--headless",
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "sub-project E" in combined, combined


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
