import typing as T

import gym
import numpy as np

from .environment import Environment


class SpaceInvaders(Environment):
    def __init__(self):
        self.env: gym.Env = gym.make("SpaceInvaders-v0")

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def step(self, action: int) -> T.Tuple[np.ndarray, float, bool]:
        s, r, f, _ = self.env.step(action)
        return s, r, f

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()
