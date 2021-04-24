import typing as T
import gym
import numpy as np

from .environment import Environment


class CartPoleFrames(Environment):
    def __init__(self):
        self.env: gym.Env = gym.make('CartPole-v1').unwrapped
        self.env.reset()

    def get_observation_space(self) -> T.Tuple[int]:
        screen: np.ndarray = self.env.render(mode="rgb_array")
        return screen.shape

    def get_action_space(self) -> T.Tuple[int]:
        return tuple(range(self.env.action_space.n))

    def reset(self) -> np.ndarray:
        self.env.reset()
        return self.env.render(mode="rgb_array").copy()

    def do_step(self, action: int) -> T.Tuple[np.ndarray, float, bool]:
        _, r, f, _ = self.env.step(action)
        return self.env.render(mode="rgb_array").copy(), r, f

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()
