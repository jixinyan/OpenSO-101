import argparse
import json
from pathlib import Path

import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from openso101.rl.bounded_policy import BoundedActorCritic
from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("state", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
state = torch.load(args.state, map_location="cpu", weights_only=False)
if state.get("action_likelihood") == "gaussian_latent":
    raise ValueError("该检查脚本使用已保存的 bounded-action rollout")
storage = state["storage"]
observations = storage["observations"].flatten(0, 1)
policy = BoundedActorCritic(observations[:1], {"policy": ["policy"], "critic": ["policy"]}, 6,
                           actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64],
                           activation="elu", noise_std_type="log", actor_obs_normalization=True,
                           critic_obs_normalization=True)
policy.load_state_dict(state["model_state_dict"])
torch.set_rng_state(state["rng_state"])


def statistics(values):
    values = values.detach()
    finite = torch.isfinite(values)
    return {"shape": list(values.shape), "nonfinite_count": int((~finite).sum()),
            "minimum_finite": float(values[finite].min()) if finite.any() else None,
            "maximum_finite": float(values[finite].max()) if finite.any() else None}


records = []
for first in range(0, len(observations), 49152):
    last = min(first + 49152, len(observations))
    obs = observations[first:last]
    actions = storage["actions"].flatten(0, 1)[first:last]
    advantages = storage["advantages"].flatten()[first:last]
    old_log_prob = storage["actions_log_prob"].flatten()[first:last]
    policy.act(obs)
    distribution = TransformedDistribution(policy.distribution, [TanhTransform(cache_size=1)])
    log_prob = distribution.log_prob(actions.clamp(-1 + 1e-6, 1 - 1e-6)).sum(dim=-1)
    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()
    surrogate = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(.8, 1.2)).mean()
    entropy = policy.entropy.mean()
    gradients = {}
    for name, loss in (("surrogate", surrogate), ("entropy", entropy)):
        gradient = torch.autograd.grad(loss, policy.log_std, retain_graph=True)[0]
        gradients[name] = statistics(gradient)
    precise = TransformedDistribution(Normal(policy.distribution.loc.double(), policy.distribution.scale.double()),
                                      [TanhTransform(cache_size=1)])
    precise_log_prob = precise.log_prob(actions.clamp(-1 + 1e-6, 1 - 1e-6).double()).sum(dim=-1)
    precise_ratio = (precise_log_prob - old_log_prob.double()).exp()
    precise_surrogate = torch.maximum(-advantages * precise_ratio, -advantages * precise_ratio.clamp(.8, 1.2)).mean()
    precise_gradient = torch.autograd.grad(precise_surrogate, policy.log_std)[0]
    normalized = policy.actor_obs_normalizer(policy.get_actor_obs(obs))
    extreme = int(log_ratio.argmax())
    records.append({"first_transition": first, "last_transition": last,
                    "observations": statistics(normalized), "mean": statistics(policy.action_mean),
                    "log_prob": statistics(log_prob), "old_log_prob": statistics(old_log_prob),
                    "log_ratio": statistics(log_ratio), "ratio": statistics(ratio),
                    "advantages": statistics(advantages), "gradients": gradients,
                    "float64_ratio": statistics(precise_ratio), "float64_surrogate_gradient": statistics(precise_gradient),
                    "maximum_log_ratio_transition": first + extreme,
                    "maximum_log_ratio_advantage": float(advantages[extreme]),
                    "maximum_log_ratio_normalized_observation": normalized[extreme].detach().tolist()})
report = {"state_sha256": digest(args.state), "source_sha256": digest(Path(__file__)),
          "rollout_transitions": len(observations), "batches": records,
          "scope": "loss_gradients_from_actual_saved_rollout_and_policy_on_cpu"}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps(report, indent=2), flush=True)
