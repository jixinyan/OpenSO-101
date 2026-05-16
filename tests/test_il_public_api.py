# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Smoke tests for the `openso101.il` public API surface.

These tests confirm the modules under `openso101.il.{policies,runners,
datasets}` are real, importable wrappers (not raise-NotImplementedError
stubs), and that their function signatures route into LeRobot as
expected. They do not require LeRobot or Isaac Sim — the import-time
contract is what we are asserting.
"""

from __future__ import annotations

import pytest


def test_il_top_level_exports_resolvable():
    """Importing the top-level package surfaces every documented symbol."""
    import openso101.il as il

    for name in (
        "ACTPolicy",
        "DiffusionPolicy",
        "LeRobotDatasetHandle",
        "TrainResult",
        "load_act_policy",
        "load_diffusion_policy",
        "load_lerobot_dataset",
        "load_policy",
        "policy_class",
        "summarize_lerobot_dataset",
        "train_il_policy",
    ):
        assert hasattr(il, name), f"openso101.il is missing {name!r}"


def test_policies_are_proxies_not_stubs():
    """ACTPolicy / DiffusionPolicy must be real proxies, not NotImplemented stubs."""
    from openso101.il.policies import ACTPolicy, DiffusionPolicy

    # The proxies' `__new__` is what defers instantiation to LeRobot.
    # If someone has reverted us to a NotImplementedError skeleton, the
    # class would either be missing or its __new__ would unconditionally
    # raise on construction *before* trying to import LeRobot.
    assert hasattr(ACTPolicy, "__new__")
    assert hasattr(DiffusionPolicy, "__new__")
    # Calling them WILL fail in CI (no LeRobot) but with an ImportError
    # from the lazy import, not a NotImplementedError from a placeholder.
    with pytest.raises((ImportError, ModuleNotFoundError)):
        ACTPolicy()
    with pytest.raises((ImportError, ModuleNotFoundError)):
        DiffusionPolicy()


def test_train_result_dataclass_contract():
    """TrainResult must expose the documented helpers."""
    from pathlib import Path

    from openso101.il.runners import TrainResult

    result = TrainResult(returncode=0, output_dir=Path("/tmp/x"), command=("a",))
    assert result.succeeded is True
    assert result.last_checkpoint == Path("/tmp/x/checkpoints/last/pretrained_model")

    failed = TrainResult(returncode=1, output_dir=Path("/tmp/y"), command=("b",))
    assert failed.succeeded is False


def test_train_il_policy_rejects_bad_dataset_dir(tmp_path, capsys):
    """A non-existent local path falls through to the Hub repo_id branch."""
    from openso101.il.runners import train_il_policy

    # We don't actually want to launch a subprocess in CI; the subprocess
    # call will fail when the python interpreter can't find `lerobot.
    # scripts.train`. We capture the printed command line to assert
    # routing.
    result = train_il_policy(
        policy="act",
        dataset="user/some-repo-that-does-not-exist",
        output_dir=str(tmp_path / "out"),
    )
    captured = capsys.readouterr()
    assert "lerobot.scripts.train" in captured.out
    assert "--policy.type=act" in captured.out
    assert "--dataset.repo_id=user/some-repo-that-does-not-exist" in captured.out
    # Subprocess failed (no lerobot in CI) — that's expected; the routing
    # is what we're testing.
    assert isinstance(result.returncode, int)


def test_load_lerobot_dataset_rejects_unknown_local_dir(tmp_path):
    """A directory without meta/info.json should raise FileNotFoundError."""
    from openso101.il.datasets import load_lerobot_dataset

    fake_dir = tmp_path / "not-a-dataset"
    fake_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_lerobot_dataset(fake_dir)


def test_load_policy_rejects_missing_path(tmp_path):
    """`load_policy` must FileNotFoundError before trying to import LeRobot."""
    from openso101.il.policies import load_policy

    with pytest.raises(FileNotFoundError):
        load_policy(tmp_path / "nope")


def test_load_policy_walks_pretrained_layout(tmp_path):
    """If a `pretrained_model/config.json` exists, the resolver finds it."""
    from openso101.il.policies.factory import _resolve_checkpoint_dir

    inner = tmp_path / "pretrained_model"
    inner.mkdir(parents=True)
    (inner / "config.json").write_text("{}")
    assert _resolve_checkpoint_dir(tmp_path) == inner


def test_load_policy_walks_checkpoints_last_layout(tmp_path):
    """The `checkpoints/last/pretrained_model` layout must also resolve."""
    from openso101.il.policies.factory import _resolve_checkpoint_dir

    inner = tmp_path / "checkpoints" / "last" / "pretrained_model"
    inner.mkdir(parents=True)
    (inner / "config.json").write_text("{}")
    assert _resolve_checkpoint_dir(tmp_path) == inner
