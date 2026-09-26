# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import pytest
import torch
from pydantic import ValidationError

from openso101.rl.config import CheckpointMeta, TrainCfg, digest


def test_training_config_rejects_incompatible_backend():
    with pytest.raises(ValidationError, match="SAC 和 TQC"):
        TrainCfg(backend="skrl", algo="sac")
    assert TrainCfg(backend="sb3", algo="tqc").algo == "tqc"
    with pytest.raises(ValueError, match="整除"):
        TrainCfg(rollout_steps=5).batch_size(1)
    assert TrainCfg(rollout_steps=16).batch_size(2) == 8


def test_checkpoint_requires_complete_unchanged_files(tmp_path):
    path = tmp_path / "model.pt"
    layer = torch.nn.Linear(3, 2)
    torch.save(layer.state_dict(), path)
    cfg = TrainCfg(iterations=1)
    meta = CheckpointMeta(task_id="OpenSO101-Lift-v0", config=cfg, git_sha="a" * 40,
                          checkpoint="model.pt", files={"model.pt": digest(path)}, completed_transitions=96)
    meta.write(tmp_path)
    assert CheckpointMeta.read(tmp_path) == meta
    torch.save(torch.nn.Linear(4, 2).state_dict(), path)
    with pytest.raises(ValueError, match="验证失败"):
        CheckpointMeta.read(tmp_path)
