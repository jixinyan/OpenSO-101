import hashlib
import math

import torch
from torch.utils.data._utils.collate import default_collate
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors


def validate_model_graph(cfg, dataset):
    if cfg.policy.device != "cpu":
        raise ValueError("IL 模型程序检查需要使用 CPU")
    if cfg.seed is None:
        raise ValueError("IL 模型程序检查需要指定 seed")
    torch.manual_seed(cfg.seed)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    if not math.isfinite(cfg.optimizer.lr):
        raise ValueError("IL optimizer lr 必须为有限数值")
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    preprocessor, postprocessor = make_pre_post_processors(cfg.policy, dataset_stats=dataset.meta.stats)
    raw_batch = default_collate([dataset[0]])
    batch = preprocessor(raw_batch)
    policy.train()
    policy.zero_grad(set_to_none=True)
    loss, losses = policy(batch)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("IL 模型 loss 必须为有限标量")
    loss.backward()
    gradient_parameters, nonzero_gradients, maximum_gradient = 0, 0, 0.0
    for parameter in policy.parameters():
        if parameter.device.type != "cpu":
            raise ValueError("IL 模型参数需要全部位于 CPU")
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise ValueError("IL 模型 gradient 包含非有限数值")
        gradient_parameters += parameter.numel()
        magnitude = float(parameter.grad.abs().max())
        nonzero_gradients += int(magnitude > 0)
        maximum_gradient = max(maximum_gradient, magnitude)
    if gradient_parameters == 0 or nonzero_gradients == 0:
        raise ValueError("IL 模型没有产生实际 gradient")
    if optimizer.state:
        raise ValueError("IL 模型程序检查不能执行 optimizer 更新")
    current = {key: (raw_batch[key][:, -1] if cfg.policy.n_obs_steps > 1 else raw_batch[key])
               for key in cfg.policy.input_features}
    observation = preprocessor(current)
    policy.eval()
    policy.reset()
    with torch.inference_mode():
        action = postprocessor(policy.select_action(observation))
    if tuple(action.shape) != (1, 6) or not torch.isfinite(action).all():
        raise ValueError("IL 模型推理需要产生六个有限关节值")
    if losses is not None and any(not math.isfinite(float(value)) for value in losses.values()):
        raise ValueError("IL 模型 loss 分量包含非有限数值")
    digest = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return {"status": "actual_cpu_model_graph_verified", "initialization": "lerobot_configured_model",
            "seed": cfg.seed, "batch_size": 1, "sampled_frame_index": 0,
            "parameters": sum(value.numel() for value in policy.parameters()),
            "gradient_parameters": gradient_parameters, "nonzero_gradient_tensors": nonzero_gradients,
            "maximum_gradient": maximum_gradient, "loss": float(loss.detach()), "loss_components": losses,
            "inference_action": action.tolist(), "model_state_sha256": digest.hexdigest(),
            "optimizer": type(optimizer).__name__,
            "optimizer_learning_rates": [group["lr"] for group in optimizer.param_groups],
            "scheduler": type(scheduler).__name__ if scheduler is not None else None,
            "optimizer_configuration_verified": True,
            "optimizer_updates": 0, "model_forward_verified": True, "model_backward_verified": True,
            "model_inference_verified": True, "gpu_tests_started": False, "task_success_verified": False}
