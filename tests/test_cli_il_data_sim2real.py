# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Acceptance-criterion tests for the il/data/sim2real CLI groups.

Subprocess-based to verify the wired-up entry point end-to-end without
requiring Isaac Lab.
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


def test_il_train_says_sub_project_c():
    proc = _run_cli(
        "il", "train",
        "--policy", "act",
        "--dataset", "/tmp/nonexistent",
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "sub-project C" in combined, combined


def test_il_play_says_sub_project_c():
    proc = _run_cli(
        "il", "play",
        "--task", "OpenSO101-PickPlace-v0",
        "--policy-path", "/tmp/nonexistent",
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "sub-project C" in combined, combined


def test_data_generate_says_sub_project_f():
    proc = _run_cli(
        "data", "generate",
        "--source", "/tmp/nonexistent",
        "--backend", "mimicgen",
        "--output", "/tmp/out",
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "sub-project F" in combined, combined


def test_data_inspect_says_sub_project_f():
    proc = _run_cli(
        "data", "inspect",
        "--dataset", "/tmp/nonexistent",
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "sub-project F" in combined, combined


def test_sim2real_deploy_says_future():
    proc = _run_cli(
        "sim2real", "deploy",
        "--policy-path", "/tmp/nonexistent",
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "future" in combined.lower() or "deferred" in combined.lower(), combined


def test_il_record_skeleton_runs_subprocess():
    """`il record` is a skeleton — running it should exit with a NotImplementedError
    trace (exit code != 0), not a parser error."""
    proc = _run_cli(
        "il", "record",
        "--task", "OpenSO101-PickPlace-v0",
        "--leader-port", "/dev/null",
        "--leader-id", "dummy",
    )
    # NotImplementedError → uncaught exception → exit 1, and the message
    # mentions the source reference.
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "NotImplementedError" in combined or "not yet" in combined.lower(), combined
