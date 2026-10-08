import torch

from openso101.teleop.so101_mapping import batched_motor_units_to_action


def select_sim_action(policy, observation: dict, device, environments: int) -> torch.Tensor:
    state = observation["observation.state"]
    if not isinstance(state, torch.Tensor) or state.shape != (environments, 6):
        raise ValueError("IL 推理需要 [N, 6] observation.state")
    for name, value in observation.items():
        if not isinstance(value, torch.Tensor) or value.shape[0] != environments or not torch.isfinite(value).all():
            raise ValueError(f"IL 推理 observation 格式或数值无效: {name}")
    features = {name: value.to(device) for name, value in observation.items()}
    preprocessor = getattr(policy, "openso101_preprocessor", None)
    postprocessor = getattr(policy, "openso101_postprocessor", None)
    with torch.inference_mode():
        if preprocessor is not None:
            features = preprocessor(features)
        action = policy.select_action(features)
        if postprocessor is not None:
            action = postprocessor(action)
        if (not isinstance(action, torch.Tensor) or action.shape != (environments, 6)
                or not action.is_floating_point() or not torch.isfinite(action).all()):
            raise ValueError("IL 推理动作需要六个有限的关节值")
        return batched_motor_units_to_action(action.to(device))
