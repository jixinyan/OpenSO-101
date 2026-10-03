import json
import math
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDictBase


def cpu_tensors(value):
    if isinstance(value, (torch.Tensor, TensorDictBase)):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {name: cpu_tensors(item) for name, item in value.items()}
    return value


class CheckedPPO(PPO):
    def __init__(self, *args, diagnostic_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostic_dir = Path(diagnostic_dir) if diagnostic_dir else None
        self.update_count = 0
        self.gradient_steps = 0
        self.latest_observation = None
        for name, parameter in self.policy.named_parameters():
            parameter.register_hook(lambda gradient, name=name: self.check_gradient(name, gradient))
        self.optimizer.register_step_pre_hook(self.check_optimizer)

    def capture_failure(self, message, invalid_tensor=None):
        if self.diagnostic_dir is None:
            return
        folder = self.diagnostic_dir / "numerical_failure"
        if folder.exists():
            return
        folder.mkdir(parents=True, exist_ok=False)
        record = {"message": message, "update_count": self.update_count,
                  "gradient_steps": self.gradient_steps, "learning_rate": self.learning_rate,
                  "storage_step": self.storage.step if self.storage is not None else None,
                  "action_likelihood": "gaussian_latent"}
        (folder / "failure.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        torch.save({"action_likelihood": "gaussian_latent",
                    "model_state_dict": cpu_tensors(self.policy.state_dict()),
                    "optimizer_state_dict": cpu_tensors(self.optimizer.state_dict()),
                    "gradients": {name: cpu_tensors(parameter.grad) for name, parameter in self.policy.named_parameters()},
                    "storage": {name: cpu_tensors(value) for name, value in vars(self.storage).items()
                                if isinstance(value, (torch.Tensor, TensorDictBase))},
                    "latest_observation": cpu_tensors(self.latest_observation),
                    "invalid_tensor": cpu_tensors(invalid_tensor),
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                   folder / "state.pt")

    def require_finite(self, name, value):
        if isinstance(value, TensorDictBase):
            for key, item in value.items():
                self.require_finite(f"{name}/{key}", item)
        elif not torch.isfinite(value).all():
            message = f"PPO 的 {name} 含有无效数值"
            self.capture_failure(message, invalid_tensor=value)
            raise FloatingPointError(message)

    def check_gradient(self, name, gradient):
        self.require_finite(f"gradient/{name}", gradient)
        return gradient

    def check_optimizer(self, optimizer, args, kwargs):
        for name, parameter in self.policy.named_parameters():
            self.require_finite(f"parameter/{name}", parameter)
            if parameter.grad is not None:
                self.require_finite(f"clipped_gradient/{name}", parameter.grad)
        self.gradient_steps += 1

    def act(self, obs):
        self.latest_observation = obs
        self.require_finite("observation", obs)
        action = self.policy.act(obs)
        self.transition.actions = self.policy.last_latent_action.detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        self.require_finite("environment_action", action)
        for name in ("actions", "values", "actions_log_prob", "action_mean", "action_sigma"):
            self.require_finite(name, getattr(self.transition, name))
        return action

    def init_storage(self, *args, **kwargs):
        super().init_storage(*args, **kwargs)
        self.storage.actions_log_prob = self.storage.actions_log_prob.double()

    def process_env_step(self, obs, rewards, dones, extras):
        self.latest_observation = obs
        self.require_finite("next_observation", obs)
        self.require_finite("reward", rewards)
        return super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs):
        self.require_finite("last_observation", obs)
        super().compute_returns(obs)
        for name in ("returns", "advantages", "values", "actions_log_prob"):
            self.require_finite(name, getattr(self.storage, name))
        from torch.distributions import Normal

        expected = Normal(self.storage.mu.double(), self.storage.sigma.double()).log_prob(
            self.storage.actions.double()).sum(dim=-1)
        error = (expected - self.storage.actions_log_prob.squeeze(-1)).abs().max()
        if error > 1e-8:
            raise RuntimeError("rollout 的实际 latent、Gaussian 参数和 log probability 不一致")
        if self.diagnostic_dir is not None:
            record = {"update": self.update_count + 1, "action_likelihood": "gaussian_latent",
                      "log_probability_dtype": str(self.storage.actions_log_prob.dtype),
                      "log_probability_maximum_error": float(error),
                      "latent_minimum": float(self.storage.actions.min()),
                      "latent_maximum": float(self.storage.actions.max()),
                      "transitions": self.storage.num_envs * self.storage.num_transitions_per_env}
            with (self.diagnostic_dir / "latent_rollouts.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")

    def update(self):
        self.update_count += 1
        result = super().update()
        if any(not math.isfinite(value) for value in result.values()):
            self.capture_failure("PPO loss 含有无效数值")
            raise FloatingPointError("PPO loss 含有无效数值")
        if self.diagnostic_dir is not None:
            record = {"update": self.update_count, "gradient_steps": self.gradient_steps,
                      "learning_rate": self.learning_rate, "losses": result}
            with (self.diagnostic_dir / "updates.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        return result
