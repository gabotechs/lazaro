import lazaro as lz
from .snake import Snake
from .apple import Apple
import typing as T
import numpy as np
import random
import time


class SnakeEnv(lz.environments.Environment):
    CHANNELS = 4
    SHAPE = (5, 5)
    RENDER_MAP = {
        0: "..",
        1: "[]",
        2: "{}",
        3: ")("
    }
    ACTION_TO_DIRECTION = {
        0: (0, -1),
        1: (-1, 0),
        2: (0, 1),
        3: (1, 0)
    }
    REWARD_CRASH = -1
    REWARD_EAT = 10
    HUNGER_LIMIT = 20

    def __init__(self):
        self.visualize = True
        self.snake = Snake()
        self.apple = Apple()
        self.eaten = 0
        self.hunger = 0
        self.steps = 0
        self.max_score = 0

    def get_observation_space(self) -> T.Tuple[int, ...]:
        return self.SHAPE

    def get_action_space(self) -> T.Tuple[int, ...]:
        return 0, 1, 2, 3

    def _render_state(self):
        state = [[0 for _ in range(self.SHAPE[0])] for _ in range(self.SHAPE[1])]
        if self.snake.position:
            snake_head = self.snake.position
            state[snake_head[0]][snake_head[1]] = Snake.HEAD
            snake_point = snake_head
            for tail_direction in self.snake.tail:
                snake_point = snake_point[0]-tail_direction[0], snake_point[1]-tail_direction[1]
                state[snake_point[0]][snake_point[1]] = Snake.TAIL
        if self.apple.position:
            state[self.apple.position[0]][self.apple.position[1]] = Apple.APPLE

        return state

    def reset(self) -> np.ndarray:
        self.eaten = 0
        self.hunger = 0
        self.steps = 0
        self.snake.clear_tail()
        self.snake.set_position(self._random_position())
        self.apple.set_position(self._random_position())
        return np.array(self._render_state())

    def _random_position(self):
        state = self._render_state()
        new_position = (random.randint(0, self.SHAPE[0] - 1), random.randint(0, self.SHAPE[1] - 1))
        while True:
            if state[new_position[0]][new_position[1]] == 0:
                return new_position
            new_position = (random.randint(0, self.SHAPE[0] - 1), random.randint(0, self.SHAPE[1] - 1))

    def do_step(self, action: int) -> T.Tuple[np.ndarray, float, bool]:
        self.steps += 1
        new_snake_position = self.ACTION_TO_DIRECTION[action][0]+self.snake.position[0], self.ACTION_TO_DIRECTION[action][1]+self.snake.position[1]

        if not (0 <= new_snake_position[0] < self.SHAPE[0] and 0 <= new_snake_position[1] < self.SHAPE[1]):
            return np.array(self._render_state()), self.REWARD_CRASH, True
        prev_state = self._render_state()
        new_cell_content = prev_state[new_snake_position[0]][new_snake_position[1]]
        if new_cell_content == Snake.TAIL:
            return np.array(prev_state), self.REWARD_CRASH, True
        elif new_cell_content == Apple.APPLE:
            self.snake.grow_tail(self.ACTION_TO_DIRECTION[action])
            self.snake.set_position(new_snake_position)
            self.apple.set_position(self._random_position())
            self.eaten += 1
            self.max_score = max(self.max_score, self.eaten)
            self.hunger -= self.HUNGER_LIMIT
            return np.array(self._render_state()), self.REWARD_EAT, False
        else:
            self.snake.shift_tail(self.ACTION_TO_DIRECTION[action])
            self.snake.set_position(new_snake_position)
            self.hunger += 1
            if self.hunger > self.HUNGER_LIMIT:
                return np.array(self._render_state()), self.REWARD_CRASH, True
            else:
                return np.array(self._render_state()), 0, False

    def render(self) -> None:
        if not self.visualize:
            return
        p = "="*self.SHAPE[1]*2+"\n"
        p += f"max score: {self.max_score}\n"
        p += f"score: {self.eaten} | steps: {self.steps}\n"
        for row in self._render_state():
            for cell in row:
                p += self.RENDER_MAP[cell]
            p += "\n"
        print(p)
        time.sleep(0.001)

    def close(self) -> None:
        pass
