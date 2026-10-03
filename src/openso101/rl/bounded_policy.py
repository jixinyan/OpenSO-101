import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner


class BoundedActorCritic(ActorCritic):
    def act(self, obs, **kwargs):
        observation = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self.update_distribution(observation)
        return self.distribution.sample().tanh().clamp(-1 + 1e-6, 1 - 1e-6)

    def act_inference(self, obs):
        return super().act_inference(obs).tanh()

    def get_actions_log_prob(self, actions):
        transformed = TransformedDistribution(self.distribution, [TanhTransform(cache_size=1)])
        return transformed.log_prob(actions.clamp(-1 + 1e-6, 1 - 1e-6)).sum(dim=-1)

    @property
    def entropy(self):
        # 使用变换后的分布计算 entropy；KL 使用原始 Gaussian 的参数。
        latent = self.distribution.rsample()
        jacobian = TanhTransform().log_abs_det_jacobian(latent, latent.tanh())
        return (self.distribution.entropy() + jacobian).sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.actor(obs)
        std = self.log_std.clamp(-5., 2.).exp().expand_as(mean)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise RuntimeError("actor 产生无效的 Gaussian 参数")
        self.distribution = Normal(mean, std)


class BoundedOnPolicyRunner(OnPolicyRunner):
    def _construct_algorithm(self, obs):
        policy_cfg = dict(self.policy_cfg)
        algorithm_cfg = dict(self.alg_cfg)
        if policy_cfg.pop("class_name") != "BoundedActorCritic" or algorithm_cfg.pop("class_name") != "PPO":
            raise ValueError("bounded runner 需要 BoundedActorCritic 与 PPO")
        policy = BoundedActorCritic(obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg).to(self.device)
        algorithm = PPO(policy, device=self.device, multi_gpu_cfg=self.multi_gpu_cfg, **algorithm_cfg)
        algorithm.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return algorithm


def runner_class(config):
    name = config["policy"]["class_name"]
    if name == "BoundedActorCritic":
        return BoundedOnPolicyRunner
    if name in ("ActorCritic", "ActorCriticRecurrent"):
        return OnPolicyRunner
    raise ValueError(f"不支持的 policy：{name}")
