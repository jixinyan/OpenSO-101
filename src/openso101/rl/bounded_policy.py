import torch
from torch.distributions import Normal, TanhTransform
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner

from .checked_ppo import CheckedPPO


class BoundedActorCritic(ActorCritic):
    def __init__(self, *args, initial_action_mean=None, freeze_demonstration_normalization=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_demonstration_normalization = freeze_demonstration_normalization
        self._normalization_reference = None
        if initial_action_mean is not None:
            mean = torch.as_tensor(initial_action_mean, dtype=self.actor[-1].bias.dtype)
            if mean.shape != self.actor[-1].bias.shape or not torch.isfinite(mean).all() or (mean.abs() >= 1).any():
                raise ValueError("初始动作均值需要有效的归一化关节位置")
            torch.nn.init.orthogonal_(self.actor[-1].weight, gain=.01)
            with torch.no_grad():
                self.actor[-1].bias.copy_(mean.atanh())

    def update_normalization(self, obs):
        if not self.freeze_demonstration_normalization:
            super().update_normalization(obs)

    def initialize_normalization_reference(self, obs):
        if not self.freeze_demonstration_normalization:
            raise ValueError("固定归一化初始化需要对应配置")
        super().update_normalization(obs)
        self.pin_normalization_reference()

    def pin_normalization_reference(self):
        if not self.freeze_demonstration_normalization:
            raise ValueError("固定归一化记录需要对应配置")
        normalizers = {"actor": self.actor_obs_normalizer, "critic": self.critic_obs_normalizer}
        self._normalization_reference = {}
        for name, normalizer in normalizers.items():
            state = normalizer.state_dict()
            if not state or float(state["count"]) <= 0 or any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError("固定归一化需要已初始化的实际统计")
            self._normalization_reference[name] = {key: value.detach().clone() for key, value in state.items()}

    def verify_normalization_reference(self):
        if self._normalization_reference is None:
            raise RuntimeError("固定归一化缺少实际统计记录")
        for name, normalizer in (("actor", self.actor_obs_normalizer), ("critic", self.critic_obs_normalizer)):
            if any(not torch.equal(value, self._normalization_reference[name][key])
                   for key, value in normalizer.state_dict().items()):
                raise RuntimeError("训练期间的固定归一化统计发生改变")
        return True

    def act(self, obs, **kwargs):
        observation = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self.update_distribution(observation)
        self.last_latent_action = self.distribution.sample()
        return self.last_latent_action.tanh()

    def act_inference(self, obs):
        return super().act_inference(obs).tanh()

    def get_actions_log_prob(self, actions):
        # rollout 保存 Gaussian latent；Tanh 的 Jacobian 在 PPO ratio 中消去。
        distribution = Normal(self.distribution.loc.double(), self.distribution.scale.double())
        return distribution.log_prob(actions.double()).sum(dim=-1)

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
    def save(self, path, infos=None):
        details = dict(infos or {})
        demonstrations = getattr(self.alg, "demonstration_updates", None)
        if demonstrations is not None:
            details["demonstration_optimizer"] = demonstrations.optimizer.state_dict()
            details["demonstration_gradient_steps"] = demonstrations.steps
        if self.alg.policy.freeze_demonstration_normalization:
            details["normalization_reference_verified"] = self.alg.policy.verify_normalization_reference()
        return super().save(path, infos=details)

    def _construct_algorithm(self, obs):
        policy_cfg = dict(self.policy_cfg)
        algorithm_cfg = dict(self.alg_cfg)
        if policy_cfg.pop("class_name") != "BoundedActorCritic" or algorithm_cfg.pop("class_name") != "PPO":
            raise ValueError("bounded runner 需要 BoundedActorCritic 与 PPO")
        policy = BoundedActorCritic(obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg).to(self.device)
        algorithm = CheckedPPO(policy, diagnostic_dir=self.cfg.get("diagnostic_dir"),
                               device=self.device, multi_gpu_cfg=self.multi_gpu_cfg, **algorithm_cfg)
        algorithm.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return algorithm


def runner_class(config):
    name = config["policy"]["class_name"]
    if name == "BoundedActorCritic":
        return BoundedOnPolicyRunner
    if name in ("ActorCritic", "ActorCriticRecurrent"):
        return OnPolicyRunner
    raise ValueError(f"不支持的 policy：{name}")
