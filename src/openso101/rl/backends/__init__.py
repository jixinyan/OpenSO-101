# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from importlib import import_module
from pathlib import Path
from typing import Protocol

from openso101.rl.config import TrainCfg


class RLBackend(Protocol):
    def train(self, env, cfg: TrainCfg, output: Path, resume: Path | None = None) -> Path: ...
    def load(self, env, folder: Path): ...


def get_backend(name: str) -> RLBackend:
    if name not in {"rsl_rl", "sb3", "skrl", "rl_games"}:
        raise ValueError(f"未知 backend：{name}")
    return import_module(f".{name}", __name__).Backend()
