import json

import torch


def record_initial_std(output, config, standard_deviation, *, resumed):
    values = standard_deviation.detach().cpu()
    if not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError("策略初始化需要有效的 Gaussian standard deviation")
    if not resumed and not torch.allclose(values, torch.full_like(values, config.initial_noise_std), atol=1e-6, rtol=1e-6):
        raise ValueError("实际策略分布与 initial_noise_std 不一致")
    report = {"backend": config.backend, "algo": config.algo,
              "distribution_scope": "gaussian_latent_standard_deviation",
              "requested_initial_std": config.initial_noise_std,
              "actual_std": values.tolist(), "resumed": resumed}
    (output / "policy_initialization.json").write_text(json.dumps(report, indent=2) + "\n")
