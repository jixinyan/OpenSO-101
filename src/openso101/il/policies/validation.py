import hashlib
import json
from pathlib import Path

import torch
from lerobot.processor import DeviceProcessorStep
from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME

from openso101.il.policies.factory import load_policy
from openso101.scenes.models import file_digest


def model_state_digest(policy):
    digest = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _inference(policy, preprocessor, postprocessor, current, seed):
    torch.manual_seed(seed)
    policy.eval()
    policy.reset()
    observation = preprocessor(current)
    if any(value.device.type != "cpu" for value in observation.values() if isinstance(value, torch.Tensor)):
        raise ValueError("checkpoint observation 需要位于 CPU")
    with torch.inference_mode():
        action = postprocessor(policy.select_action(observation))
    if tuple(action.shape) != (1, 6) or action.device.type != "cpu" or not torch.isfinite(action).all():
        raise ValueError("checkpoint 推理需要六个有限 CPU 关节值")
    return action


def _check_processor_states(original, loaded):
    if len(original.steps) != len(loaded.steps):
        raise ValueError("checkpoint processor 数量不一致")
    tensors = 0
    for reference, restored in zip(original.steps, loaded.steps, strict=True):
        if type(reference) is not type(restored):
            raise ValueError("checkpoint processor 类型不一致")
        if isinstance(restored, DeviceProcessorStep) and restored.tensor_device.type != "cpu":
            raise ValueError("checkpoint processor device 需要为 CPU")
        expected, actual = reference.state_dict(), restored.state_dict()
        if expected.keys() != actual.keys():
            raise ValueError("checkpoint processor 状态字段不一致")
        for name, value in actual.items():
            if value.device.type != "cpu" or not torch.equal(expected[name], value):
                raise ValueError(f"checkpoint processor 状态不一致: {name}")
            tensors += 1
    return tensors


def validate_checkpoint(policy, preprocessor, postprocessor, current, output: Path, seed: int):
    if output.exists() or policy.config.device != "cpu":
        raise ValueError("checkpoint 检查需要 CPU 模型及新目录")
    reference_digest = model_state_digest(policy)
    reference_action = _inference(policy, preprocessor, postprocessor, current, seed)
    policy.config.device = "cuda:0"
    try:
        policy.save_pretrained(output)
    finally:
        policy.config.device = "cpu"
    preprocessor.save_pretrained(output, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
    postprocessor.save_pretrained(output, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")
    files = {str(path.relative_to(output)): file_digest(path) for path in output.rglob("*") if path.is_file()}
    with (output / "config.json").open() as stream:
        saved_device = json.load(stream)["device"]
    if saved_device != "cuda:0":
        raise ValueError("checkpoint 需要保存计划使用的 CUDA 配置")
    loaded = load_policy(output, device="cpu")
    if loaded.config.device != "cpu" or any(value.device.type != "cpu" for value in loaded.parameters()):
        raise ValueError("重新加载的 checkpoint 参数需要全部位于 CPU")
    if model_state_digest(loaded) != reference_digest:
        raise ValueError("重新加载的模型状态 SHA256 不一致")
    processor_tensors = _check_processor_states(preprocessor, loaded.openso101_preprocessor)
    processor_tensors += _check_processor_states(postprocessor, loaded.openso101_postprocessor)
    action = _inference(loaded, loaded.openso101_preprocessor, loaded.openso101_postprocessor, current, seed)
    error = float((reference_action - action).abs().max())
    if error > 1e-6:
        raise ValueError("重新加载的模型推理结果不一致")
    if {str(path.relative_to(output)): file_digest(path) for path in output.rglob("*") if path.is_file()} != files:
        raise ValueError("checkpoint 加载期间文件 SHA256 发生变化")
    return {"status": "actual_cpu_checkpoint_roundtrip_verified", "directory": str(output),
            "initialization": "lerobot_configured_model", "saved_config_device": saved_device,
            "requested_device": "cpu", "model_state_sha256": reference_digest,
            "processor_state_tensors": processor_tensors, "files": files,
            "maximum_action_error": error, "inference_action": action.tolist(),
            "checkpoint_files_unchanged": True, "optimizer_updates": 0,
            "gpu_tests_started": False, "task_success_verified": False}
