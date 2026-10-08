# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _resolve_checkpoint_dir(path: str | Path) -> Path:
    """读取本地模型、训练目录或经过官方检查的 Hub repo_id。"""
    local = Path(path).expanduser()
    if not local.exists():
        if isinstance(path, Path) or str(path).startswith(("/", ".", "~")) or len(local.parts) != 2:
            raise FileNotFoundError(f"模型目录不存在: {local}")
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import validate_repo_id

        validate_repo_id(str(path))
        local = Path(snapshot_download(repo_id=str(path), repo_type="model"))
    if not local.is_dir():
        raise ValueError(f"模型来源需要为目录: {local}")
    root = local.resolve()
    for candidate in (root, root / "pretrained_model", root / "checkpoints/last/pretrained_model"):
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError(f"模型目录缺少 config.json: {root}")


def validate_checkpoint_files(root: Path) -> dict:
    """检查完整模型和 processors 的实际文件。"""
    from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
    from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME

    if not (root / SAFETENSORS_SINGLE_FILE).is_file():
        raise FileNotFoundError(f"模型权重不存在: {root / SAFETENSORS_SINGLE_FILE}")
    configurations = {}
    for name in (POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME):
        with (root / f"{name}.json").open() as stream:
            configuration = json.load(stream)
        for step in configuration["steps"]:
            if "state_file" in step and not (root / step["state_file"]).is_file():
                raise FileNotFoundError(f"processor 状态文件不存在: {root / step['state_file']}")
        configurations[name] = configuration
    return configurations


def _processor_overrides(configuration: dict, device: str) -> dict:
    device_steps = {"device_processor", "normalizer_processor", "unnormalizer_processor"}
    return {step["registry_name"]: {"device": device}
            for step in configuration["steps"] if step.get("registry_name") in device_steps}


def policy_class(name: str) -> type:
    """通过 LeRobot factory 读取算法类。"""
    from lerobot.policies.factory import get_policy_class

    return get_policy_class(name)


def load_policy(
    path: str | Path,
    *,
    device: str | None = None,
) -> Any:
    """读取实际权重、配置与 observation/action processors，返回推理模型。"""
    local = Path(path).expanduser()
    if (local / "student.json").is_file():
        from openso101.rl.student import RLStudentPolicy
        from openso101.rl.gpu_guard import verify_cuda_inference

        verify_cuda_inference(device or "cpu")
        return RLStudentPolicy(local.resolve(), device or "cpu")
    ckpt_dir = _resolve_checkpoint_dir(path)
    processors = validate_checkpoint_files(ckpt_dir)

    import lerobot.policies  # noqa: F401  注册实际 policy 配置类型。

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME
    from lerobot.utils.utils import is_torch_device_available
    import torch

    requested = str(torch.device(device)) if device is not None else None
    if requested is not None and not is_torch_device_available(requested):
        raise ValueError(f"请求的模型 device 不可用: {requested}")
    cfg = PreTrainedConfig.from_pretrained(
        ckpt_dir, cli_overrides=[f"--device={requested}"] if requested is not None else [])
    if requested is not None and cfg.device != requested:
        raise ValueError("模型配置 device 与请求不一致")
    from openso101.rl.gpu_guard import verify_cuda_inference

    verify_cuda_inference(cfg.device)
    pre_overrides = _processor_overrides(processors[POLICY_PREPROCESSOR_DEFAULT_NAME], cfg.device)
    post_overrides = _processor_overrides(processors[POLICY_POSTPROCESSOR_DEFAULT_NAME], "cpu")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(ckpt_dir),
        preprocessor_overrides=pre_overrides, postprocessor_overrides=post_overrides)
    cls = policy_class(cfg.type)
    policy = cls.from_pretrained(ckpt_dir, config=cfg, strict=True)
    policy.eval()
    policy.openso101_preprocessor = preprocessor
    policy.openso101_postprocessor = postprocessor
    return policy


__all__ = ["load_policy", "policy_class"]
