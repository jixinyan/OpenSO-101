from math import sqrt
from statistics import NormalDist

import torch


def episode_quotas(episodes: int, environments: int, device):
    if (isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1
            or isinstance(environments, bool) or not isinstance(environments, int) or environments < 1):
        raise ValueError("episode 数量和环境数量必须为正整数")
    quotas = torch.full((environments,), episodes // environments, dtype=torch.int64, device=device)
    quotas[:episodes % environments] += 1
    return quotas


def success_interval(successes: int, episodes: int, confidence: float = 0.95):
    if (isinstance(successes, bool) or not isinstance(successes, int)
            or isinstance(episodes, bool) or not isinstance(episodes, int)
            or episodes < 1 or not 0 <= successes <= episodes
            or not 0 < confidence < 1):
        raise ValueError("成功数量、episode 数量或 confidence 无效")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    fraction = successes / episodes
    denominator = 1 + z * z / episodes
    center = (fraction + z * z / (2 * episodes)) / denominator
    radius = z * sqrt(fraction * (1 - fraction) / episodes + z * z / (4 * episodes * episodes)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)
