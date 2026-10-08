import torch

from openso101.rl.evaluation import episode_quotas, success_interval


class CohortEvaluation:
    def __init__(self, episodes: int, environments: int, device):
        self.quotas = episode_quotas(episodes, environments, device)
        self.completed = torch.zeros_like(self.quotas)
        self.active = torch.zeros(environments, dtype=torch.bool, device=device)
        self.previous_steps = torch.zeros_like(self.quotas)
        self.episodes = episodes
        self.records = []
        self.cohorts = 0

    @property
    def finished(self) -> bool:
        return len(self.records) == self.episodes

    @property
    def cohort_finished(self) -> bool:
        return not bool(self.active.any())

    def begin_cohort(self) -> None:
        if self.finished or not self.cohort_finished:
            raise ValueError("新的评估批次需要等待当前 episode 全部结束，并且仍有待执行的 episode")
        self.active.copy_(self.completed < self.quotas)
        self.previous_steps.zero_()
        self.cohorts += 1

    def consume(self, success, terminated, truncated, steps) -> None:
        if not self.cohorts or self.cohort_finished:
            raise ValueError("评估步骤需要一个正在运行的批次")
        for name, value, dtype in (("success", success, torch.bool),
                                   ("terminated", terminated, torch.bool),
                                   ("truncated", truncated, torch.bool),
                                   ("steps", steps, torch.int64)):
            if (not isinstance(value, torch.Tensor) or value.shape != self.active.shape
                    or value.dtype != dtype or value.device != self.active.device):
                raise ValueError(f"评估步骤的 Tensor 格式不一致: {name}")
        done = terminated | truncated
        if bool((success & ~done).any()):
            raise ValueError("成功记录需要包含当前步骤的 episode 结束状态")
        if bool((self.active & (steps != self.previous_steps + 1)).any()):
            raise ValueError("正在运行的 episode 步数需要连续增加")
        for index in torch.nonzero(self.active & done, as_tuple=False).flatten().tolist():
            self.completed[index] += 1
            self.records.append({"environment": index, "cohort": self.cohorts,
                                 "episode": int(self.completed[index]), "success": bool(success[index]),
                                 "steps": int(steps[index]), "terminated": bool(terminated[index]),
                                 "truncated": bool(truncated[index])})
            self.active[index] = False
        self.previous_steps.copy_(steps)

    def report(self) -> dict:
        count = len(self.records)
        successes = [item for item in self.records if item["success"]]
        return {"n": count, "requested_episodes": self.episodes,
                "success_count": len(successes),
                "success_rate": len(successes) / count if count else None,
                "success_rate_wilson_95": success_interval(len(successes), count) if count else None,
                "mean_steps_to_success": sum(item["steps"] for item in successes) / len(successes) if successes else None,
                "episodes_per_environment": self.completed.tolist(), "episode_quotas": self.quotas.tolist(),
                "cohorts": self.cohorts, "episodes": list(self.records),
                "policy_reset_scope": "completed_cohort", "success_sampling": "post_physics_before_reset"}
