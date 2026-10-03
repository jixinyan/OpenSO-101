import json
import shutil
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from .config import CheckpointMeta, TrainCfg, digest


class TrainingRunMeta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    task_id: str
    task_profile: Literal["default", "grasp_v2", "grasp_v3", "grasp_v4"] = "default"
    config: TrainCfg
    git_sha: str
    num_envs: int = Field(gt=0)
    scene_sha256: str | None = None
    prior_transitions: int = Field(default=0, ge=0)
    start_iteration: int = Field(default=0, ge=0)
    files: dict[str, str]

    def write(self, folder):
        path = Path(folder) / "run.json"
        with path.open("x") as stream:
            stream.write(self.model_dump_json(indent=2))

    @classmethod
    def read(cls, folder):
        folder = Path(folder).resolve()
        metadata = cls.model_validate_json((folder / "run.json").read_text())
        for name, expected in metadata.files.items():
            path = (folder / name).resolve()
            if not path.is_relative_to(folder) or digest(path) != expected:
                raise ValueError(f"训练文件校验失败：{name}")
        return metadata


def snapshot(args):
    run = Path(args.run).resolve()
    metadata = TrainingRunMeta.read(run)
    if metadata.config.backend != "rsl_rl" or metadata.config.algo != "ppo":
        raise ValueError("snapshot 当前支持 RSL PPO")
    checkpoint = (run / args.checkpoint).resolve()
    if checkpoint.parent != run or checkpoint.suffix != ".pt":
        raise ValueError("checkpoint 必须为训练目录中的 .pt 文件")
    before = digest(checkpoint)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    destination = output / checkpoint.name
    shutil.copy2(checkpoint, destination)
    if digest(destination) != before or digest(checkpoint) != before:
        raise RuntimeError("checkpoint 正在写入，无法保存一致的副本")
    model = torch.load(destination, map_location="cpu", weights_only=False)
    iteration = model["iter"]
    if not isinstance(iteration, int):
        raise ValueError("checkpoint iter 必须为整数")
    completed = iteration - metadata.start_iteration + 1
    if completed <= 0 or completed > metadata.config.iterations:
        raise ValueError("checkpoint iter 超出本次训练范围")
    if not all(torch.isfinite(value).all() for value in model["model_state_dict"].values()):
        raise ValueError("checkpoint 包含无效模型数值")
    names = set(metadata.files) | {"run.json", "backend.json"}
    for name in sorted(names):
        source = run / name
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    files = {name: digest(output / name) for name in names}
    files[checkpoint.name] = before
    result = CheckpointMeta(
        task_id=metadata.task_id, task_profile=metadata.task_profile, config=metadata.config,
        git_sha=metadata.git_sha, checkpoint=checkpoint.name, files=files,
        scene_sha256=metadata.scene_sha256,
        completed_transitions=metadata.prior_transitions + completed * metadata.config.rollout_steps * metadata.num_envs,
    )
    result.write(output)
    CheckpointMeta.read(output)
    report = {"status": "checkpoint_snapshot_verified", "iteration": iteration,
              "completed_transitions": result.completed_transitions, "checkpoint_sha256": before,
              "training_git_sha": metadata.git_sha, "task_profile": metadata.task_profile}
    (output / "snapshot.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0
