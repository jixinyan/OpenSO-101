import torch
from tensordict import TensorDict


def demonstration_windows(metadata, frame_count, length, device):
    if length < 2:
        raise ValueError("连续动作监督需要至少两个步骤")
    offset, starts = 0, []
    for source in metadata["sources"]:
        for episode in source["episodes"]:
            frames = episode["frames"]
            if frames < length:
                raise ValueError("实际 episode 的长度不足以支持指定的连续监督范围")
            starts.extend(range(offset, offset + frames - length + 1))
            offset += frames
    if offset != frame_count or not starts:
        raise ValueError("连续监督需要完整且一致的实际 episode 范围")
    offset = 0
    action_slice = None
    for term in metadata["policy_observation_terms"]:
        if term["name"] == "actions":
            if action_slice is not None or term["size"] != 6:
                raise ValueError("连续监督需要唯一的六维 previous actions")
            action_slice = slice(offset, offset + term["size"])
        offset += term["size"]
    if action_slice is None:
        raise ValueError("实际观测没有记录 previous actions")
    windows = torch.tensor(starts, device=device)[:, None] + torch.arange(length, device=device)[None]
    return windows, action_slice, offset


def sequence_predictions(policy, observations, windows, action_slice):
    if windows.ndim != 2 or windows.shape[1] < 2:
        raise ValueError("连续监督需要二维实际帧索引")
    means, predictions = [], []
    previous = None
    for step in range(windows.shape[1]):
        measured = observations[windows[:, step]].clone()
        if previous is not None:
            measured[:, action_slice] = previous
        obs = TensorDict({"policy": measured}, batch_size=[len(windows)])
        mean = policy.actor(policy.actor_obs_normalizer(policy.get_actor_obs(obs)))
        previous = mean.tanh()
        if not torch.isfinite(mean).all() or not torch.isfinite(previous).all():
            raise RuntimeError("实际模型连续前向计算产生无效数值")
        means.append(mean)
        predictions.append(previous)
    return torch.stack(means, dim=1), torch.stack(predictions, dim=1)


def sequence_loss(policy, observations, actions, windows, action_slice, *, margin, objective):
    means, predictions = sequence_predictions(policy, observations, windows, action_slice)
    targets = actions[windows]
    action_mse = torch.nn.functional.mse_loss(predictions, targets)
    if objective == "latent_mse":
        loss = torch.nn.functional.mse_loss(means, targets.clamp(-1 + margin, 1 - margin).atanh())
    elif objective == "bounded_mse":
        loss = action_mse
    else:
        raise ValueError("连续监督的 objective 无效")
    return loss, action_mse
