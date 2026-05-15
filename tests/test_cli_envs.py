# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import pytest

pytest.importorskip("isaaclab")

import openso101  # noqa: F401 (registers built-in tasks)
from openso101.cli.envs import list_envs


def test_list_envs_returns_openso101_ids():
    ids = list_envs()
    expected = {
        "OpenSO101-Lift-v0",
        "OpenSO101-PickPlace-v0",
        "OpenSO101-Stack-v0",
    }
    assert expected.issubset(set(ids)), f"missing: {expected - set(ids)}"


def test_list_envs_only_returns_openso101_prefix():
    ids = list_envs()
    for tid in ids:
        assert tid.startswith("OpenSO101-"), tid
