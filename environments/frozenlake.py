import typing as T

import gym
import numpy as np

from .environment import Environment


class FrozenLake(Environment):
    def __init__(self):
        self.visualize: bool = True
        self.env: gym.Env = gym.make("FrozenLake-v0")

    def get_observation_space(self) -> T.Tuple[int]:
        return self.env.observation_space.n,

    def get_action_space(self) -> T.Tuple[int]:
        return tuple(range(self.env.action_space.n))

    @staticmethod
    def form_state(index: int) -> np.ndarray:
        s = np.zeros(16)
        s[index] = 1
        return s

    def reset(self) -> np.ndarray:
        return self.form_state(self.env.reset())

    def do_step(self, action: int) -> T.Tuple[np.ndarray, float, bool]:
        s, r, f, _ = self.env.step(action)
        return self.form_state(s), 1 if r > 0 else (-1 if f else 0), f

    def render(self) -> None:
        if self.visualize:
            self.env.render()

    def close(self) -> None:
        self.env.close()
