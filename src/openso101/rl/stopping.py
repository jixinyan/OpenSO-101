from dataclasses import dataclass

import gymnasium as gym


@dataclass
class TrainingStopRequest:
    signal: int | None = None

    def check(self):
        if self.signal is not None:
            raise SystemExit(128 + self.signal)


class TrainingStopGuard(gym.Wrapper):
    def __init__(self, env, request):
        super().__init__(env)
        self.request = request

    def step(self, action):
        self.request.check()
        result = self.env.step(action)
        self.request.check()
        return result

    def reset(self, **kwargs):
        self.request.check()
        result = self.env.reset(**kwargs)
        self.request.check()
        return result
