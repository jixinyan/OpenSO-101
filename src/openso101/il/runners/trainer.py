# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from openso101.il.datasets.lerobot_adapter import resolve_lerobot_source
from openso101.rl.gpu_scope import gpu_scope


_SUPPORTED_POLICIES = ("act", "diffusion")
_MANAGED_ARGUMENTS = (
    "--policy.type", "--policy.device", "--policy.push_to_hub", "--policy.path",
    "--dataset.repo_id", "--dataset.root", "--output_dir", "--resume", "--config_path",
    "--steps", "--batch_size", "--wandb.enable",
)


@dataclass(frozen=True)
class TrainPlan:
    output_dir: Path
    command: tuple[str, ...]
    physical_gpu: int
    dataset_root: Path | None


@dataclass(frozen=True)
class TrainResult:
    returncode: int
    output_dir: Path
    command: tuple[str, ...]
    preparation_report: Path
    gpu_report: Path

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    @property
    def last_checkpoint(self) -> Path:
        return self.output_dir / "checkpoints" / "last" / "pretrained_model"


def _positive_count(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError(f"{name} 必须为正整数")


def build_train_plan(
    *, policy: str, dataset: str | Path, output_dir: str | Path | None = None,
    repo_id: str | None = None, steps: int | None = None, batch_size: int | None = None,
    wandb: bool = False, extra_args: Sequence[str] | None = None, gpu: int | None = None,
) -> TrainPlan:
    if policy not in _SUPPORTED_POLICIES:
        raise ValueError(f"IL policy 必须为 {_SUPPORTED_POLICIES}")
    _positive_count("steps", steps)
    _positive_count("batch_size", batch_size)
    forwarded = list(extra_args or ())
    if forwarded and forwarded[0] == "--":
        forwarded.pop(0)
    for argument in forwarded:
        if not isinstance(argument, str) or not argument.startswith("--") or "=" not in argument:
            raise ValueError("LeRobot 额外参数需要使用 --name=value")
        if argument.partition("=")[0] in _MANAGED_ARGUMENTS:
            raise ValueError(f"额外参数不能重复指定训练入口管理的参数: {argument}")
    repo = Path(__file__).resolve().parents[4]
    out = (Path(output_dir).expanduser() if output_dir is not None else
           repo / "outputs/rl_progress/il" / policy / str(time.time_ns())).resolve()
    if out.exists():
        raise FileExistsError(f"训练输出目录必须尚未存在: {out}")
    root, effective_repo_id = resolve_lerobot_source(dataset, repo_id=repo_id)
    if root is not None and out.is_relative_to(root):
        raise ValueError("训练输出目录需要位于数据集目录之外")
    scope = gpu_scope()
    physical_gpu = scope.default_gpu if gpu is None else gpu
    if isinstance(physical_gpu, bool) or not isinstance(physical_gpu, int):
        raise ValueError("gpu 必须为物理设备整数编号")
    scope.validate_allocation([physical_gpu])
    command = [sys.executable, "-m", "lerobot.scripts.lerobot_train",
               f"--policy.type={policy}", "--policy.device=cuda:0", "--policy.push_to_hub=false",
               f"--output_dir={out}", f"--dataset.repo_id={effective_repo_id}"]
    if root is not None:
        command.append(f"--dataset.root={root}")
    if steps is not None:
        command.append(f"--steps={steps}")
    if batch_size is not None:
        command.append(f"--batch_size={batch_size}")
    if wandb:
        command.append("--wandb.enable=true")
    return TrainPlan(out, tuple([*command, *forwarded]), physical_gpu, root)


def _prepare_plan(plan: TrainPlan, report_dir: str | Path | None = None) -> Path:
    repo = Path(__file__).resolve().parents[4]
    destination = (Path(report_dir).expanduser() if report_dir is not None else
                   repo / "outputs/rl_progress/il_preparation" / str(time.time_ns())).resolve()
    if not destination.is_relative_to(repo / "outputs") or destination.exists():
        raise ValueError("IL 准备检查需要位于 outputs 的新目录")
    if destination.is_relative_to(plan.output_dir):
        raise ValueError("IL 检查记录需要位于训练输出目录之外")
    if plan.dataset_root is not None and destination.is_relative_to(plan.dataset_root):
        raise ValueError("IL 检查记录需要位于数据集目录之外")
    temporary = repo / "outputs/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OPENSO101_SKIP_ISAAC": "1",
                               "TMPDIR": str(temporary)}
    command = [sys.executable, "-m", "openso101.il.runners.preparation",
               "--report-dir", str(destination), "--physical-gpu", str(plan.physical_gpu),
               "--", *plan.command[3:]]
    subprocess.run(command, check=True, cwd=repo, env=environment)
    return destination / "report.json"


def prepare_il_policy(
    *, policy: str, dataset: str | Path, output_dir: str | Path | None = None,
    repo_id: str | None = None, steps: int | None = None, batch_size: int | None = None,
    wandb: bool = False, extra_args: Sequence[str] | None = None, gpu: int | None = None,
    report_dir: str | Path | None = None,
) -> Path:
    plan = build_train_plan(policy=policy, dataset=dataset, output_dir=output_dir, repo_id=repo_id,
                            steps=steps, batch_size=batch_size, wandb=wandb, extra_args=extra_args, gpu=gpu)
    return _prepare_plan(plan, report_dir)


def train_il_policy(
    *, policy: str, dataset: str | Path, output_dir: str | Path | None = None,
    repo_id: str | None = None, steps: int | None = None, batch_size: int | None = None,
    wandb: bool = False, extra_args: Sequence[str] | None = None, gpu: int | None = None,
    check: bool = False,
) -> TrainResult:
    from openso101.rl.gpu_guard import automatic_report, run_guarded

    plan = build_train_plan(policy=policy, dataset=dataset, output_dir=output_dir, repo_id=repo_id,
                            steps=steps, batch_size=batch_size, wandb=wandb, extra_args=extra_args, gpu=gpu)
    preparation = _prepare_plan(plan)
    repo = Path(__file__).resolve().parents[4]
    gpu_report = automatic_report(repo)
    command = [sys.executable, "-m", "openso101.il.runners.worker",
               "--log", str(preparation.parent / "training.log"), "--", *plan.command]
    print(f"[openso101.il] 训练命令: {shlex.join(plan.command)}", flush=True)
    from openso101.il.policies.simulation import attach_simulation_settings, SIMULATION_SETTINGS_FILE

    try:
        returncode = run_guarded(command, plan.physical_gpu, repo, gpu_report)
    finally:
        attach_simulation_settings(plan.output_dir, preparation.parent / "pretrained_model" / SIMULATION_SETTINGS_FILE)
    result = TrainResult(returncode, plan.output_dir, plan.command, preparation, gpu_report)
    if check and not result.succeeded:
        raise RuntimeError(f"LeRobot 训练退出码为 {returncode}，GPU 记录: {gpu_report}")
    return result


__all__ = ["TrainPlan", "TrainResult", "build_train_plan", "prepare_il_policy", "train_il_policy"]
