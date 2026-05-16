# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Acceptance-criterion tests for the il + sim2real CLI groups.

Subprocess-based to verify the wired-up entry point end-to-end without
requiring Isaac Lab. Also asserts that the historical `data` subcommand
(synthetic data generation) has been removed.
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


def test_il_train_dispatches_to_lerobot_subprocess():
    """`il train` shells out to `python -m lerobot.scripts.train`. With
    LeRobot missing in the CI env the subprocess exits non-zero, but
    the wrapper must reach the subprocess call (proof that argparse +
    dispatch are wired correctly)."""
    proc = _run_cli(
        "il", "train",
        "--policy", "act",
        "--dataset", "/tmp/nonexistent",
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0
    # Either the launcher banner reached stdout, or the subprocess
    # itself printed a usage/error — both prove dispatch fired.
    assert (
        "lerobot.scripts.train" in combined
        or "LeRobot trainer" in combined
        or "Launching LeRobot trainer" in combined
    ), combined


def test_il_play_dispatches_without_isaaclab():
    """`il play` needs Isaac Sim. Without it the command should exit
    non-zero with an import-style traceback, NOT a parser error."""
    proc = _run_cli(
        "il", "play",
        "--task", "OpenSO101-PickPlace-v0",
        "--policy-path", "/tmp/nonexistent",
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    assert (
        "isaaclab" in combined.lower()
        or "AppLauncher" in combined
    ), combined


def test_data_subcommand_is_gone():
    """Synthetic data gen was removed; the `data` top-level subcommand must not exist."""
    proc = _run_cli("data", "--help")
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    assert "invalid choice" in combined.lower(), combined


def test_sim2real_deploy_requires_follower_port_and_id():
    """deploy needs --follower-port + --follower-id to know what to drive."""
    proc = _run_cli(
        "sim2real", "deploy",
        "--policy-path", "/tmp/nonexistent",
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    # argparse "required" arg failure has return code 2 and mentions
    # the missing flag.
    assert "follower-port" in combined.lower() or "follower-id" in combined.lower(), combined


def test_sim2real_deploy_dispatches_with_all_required_args():
    """All flags wired — actual hardware connect will fail (no follower
    on /dev/null), but argparse must succeed and dispatch must reach
    the deploy() function."""
    proc = _run_cli(
        "sim2real", "deploy",
        "--policy-path", "/tmp/nonexistent",
        "--follower-port", "/dev/null",
        "--follower-id", "dummy_follower",
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    # Either lerobot import failed (no LeRobot in CI) or the connect
    # failed against /dev/null. Both prove dispatch fired.
    assert (
        "lerobot" in combined.lower()
        or "follower" in combined.lower()
        or "policy" in combined.lower()
    ), combined


def test_il_record_dispatches_without_isaaclab():
    """`il record` dispatches to _cmd_record and fails informatively when
    isaaclab is absent (CI runs without Isaac Sim). The point is to catch
    argparse-level regressions, not to exercise the full record loop."""
    proc = _run_cli(
        "il", "record",
        "--task", "OpenSO101-PickPlace-v0",
        "--leader-port", "/dev/null",
        "--leader-id", "dummy",
    )
    combined = proc.stdout + proc.stderr
    # Exit non-zero AND mentions the missing isaaclab import (proof that the
    # CLI reached _cmd_record rather than failing in the parser).
    assert proc.returncode != 0, combined
    assert "isaaclab" in combined.lower() or "NotImplementedError" in combined, combined


def test_il_record_accepts_startup_sync_flags():
    """The startup home-pose hold exposes --startup-sync (opt-in) and
    --startup-sync-threshold; both should parse without error."""
    proc = _run_cli(
        "il", "record",
        "--task", "OpenSO101-PickPlace-v0",
        "--leader-port", "/dev/null",
        "--leader-id", "dummy",
        "--startup-sync",
        "--startup-sync-threshold", "0.5",
    )
    # The command will fail later (no isaaclab) but argparse must accept the
    # flags. We assert only that argparse did not reject them.
    combined = proc.stdout + proc.stderr
    assert "unrecognized arguments" not in combined.lower(), combined
    assert "--startup-sync" not in combined or "isaaclab" in combined.lower(), combined


def test_il_record_accepts_leader_async_flags():
    """--no-leader-async and --auto-save must parse cleanly."""
    proc = _run_cli(
        "il", "record",
        "--task", "OpenSO101-PickPlace-v0",
        "--leader-port", "/dev/null",
        "--leader-id", "dummy",
        "--no-leader-async",
        "--auto-save",
    )
    combined = proc.stdout + proc.stderr
    assert "unrecognized arguments" not in combined.lower(), combined
