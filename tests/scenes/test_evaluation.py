import pytest
import torch

from openso101.rl.evaluation import episode_quotas, success_interval


@pytest.mark.parametrize("episodes,environments", [(100, 64), (4, 64), (100, 1), (1, 1), (256, 64)])
def test_episode_allocation_preserves_requested_count(episodes, environments):
    quotas = episode_quotas(episodes, environments, "cpu")
    assert quotas.dtype == torch.int64
    assert quotas.shape == (environments,)
    assert int(quotas.sum()) == episodes
    assert int(quotas.max() - quotas.min()) <= 1
    assert int(quotas.min()) >= 0


@pytest.mark.parametrize("episodes,environments", [(0, 64), (100, 0), (-1, 64), (True, 64), (100, 1.5)])
def test_episode_allocation_rejects_invalid_counts(episodes, environments):
    with pytest.raises(ValueError):
        episode_quotas(episodes, environments, "cpu")


def test_success_interval_covers_zero_and_complete_success():
    low, high = success_interval(0, 100)
    assert low == 0
    assert high == pytest.approx(0.03699349820698568)
    low, high = success_interval(100, 100)
    assert low == pytest.approx(0.9630065017930143)
    assert high == 1
    low, high = success_interval(50, 100)
    assert low == pytest.approx(0.4038315303659957)
    assert high == pytest.approx(0.5961684696340044)


@pytest.mark.parametrize("successes,episodes", [(-1, 100), (101, 100), (0, 0), (True, 100)])
def test_success_interval_rejects_invalid_counts(successes, episodes):
    with pytest.raises(ValueError):
        success_interval(successes, episodes)
