import typing as T

import gym
import numpy as np

from .environment import Environment


class CartPole(Environment):
    OBSERVATION_SPACE = 4
    ACTION_SPACE = 2

    def __init__(self):
        self.visualize: bool = True
        self.env: gym.Env = gym.make("CartPole-v1")

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def step(self, action: int) -> T.Tuple[np.ndarray, float, bool]:
        s, r, f, _ = self.env.step(action)
        return s, r, f

    def render(self) -> None:
        if self.visualize:
            self.env.render()

    def close(self) -> None:
        self.env.close()
