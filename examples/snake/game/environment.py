import random
import typing as T

import lazaro as lz
from .game_objects import Snake, Apple


class SnakeEnv(lz.environments.Environment):
    CHANNELS = 4
    SHAPE = (7, 7)
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
    REWARD_EAT = 1
    HUNGER_LIMIT = 40

    def __init__(self):
        self.visualize = True
        self.snake = Snake()
        self.apple = Apple()
        self.eaten = 0
        self.hunger = 0
        self.steps = 0
        self.max_score = 0
        self.episode = 0

    def _render_state(self):
        # generate empty board
        state = [[0 for _ in range(self.SHAPE[0])] for _ in range(self.SHAPE[1])]
        # position the snake in the board
        if self.snake.position:
            snake_head = self.snake.position
            state[snake_head[0]][snake_head[1]] = Snake.HEAD
            snake_point = snake_head
            for tail_direction in self.snake.tail:
                snake_point = snake_point[0]-tail_direction[0], snake_point[1]-tail_direction[1]
                state[snake_point[0]][snake_point[1]] = Snake.TAIL
        # position the apple in the board
        if self.apple.position:
            state[self.apple.position[0]][self.apple.position[1]] = Apple.APPLE

        return state

    def _random_position(self):
        state = self._render_state()
        new_position = (random.randint(0, self.SHAPE[0] - 1), random.randint(0, self.SHAPE[1] - 1))
        # if there is something in the randomly selected position iterate until find an empty one
        while True:
            if state[new_position[0]][new_position[1]] == 0:
                return new_position
            new_position = (random.randint(0, self.SHAPE[0] - 1), random.randint(0, self.SHAPE[1] - 1))

    def reset(self) -> T.List[T.List[int]]:
        self.episode += 1
        self.eaten = 0
        self.hunger = 0
        self.steps = 0
        self.snake.clear_tail()
        self.snake.set_position(self._random_position())
        self.apple.set_position(self._random_position())
        return self._render_state()

    def step(self, action: int) -> T.Tuple[T.List[T.List[int]], float, bool]:
        self.steps += 1
        new_snake_position = (self.ACTION_TO_DIRECTION[action][0]+self.snake.position[0],
                              self.ACTION_TO_DIRECTION[action][1]+self.snake.position[1])

        # if snake has gone out of limits
        if not (0 <= new_snake_position[0] < self.SHAPE[0] and 0 <= new_snake_position[1] < self.SHAPE[1]):
            return self._render_state(), self.REWARD_CRASH, True

        # if snake has crashed with her own tail
        prev_state = self._render_state()
        new_cell_content = prev_state[new_snake_position[0]][new_snake_position[1]]
        if new_cell_content == Snake.TAIL:
            return prev_state, self.REWARD_CRASH, True

        # if snake has stepped into the apple
        elif new_cell_content == Apple.APPLE:
            self.snake.grow_tail(self.ACTION_TO_DIRECTION[action])
            self.snake.set_position(new_snake_position)
            self.apple.set_position(self._random_position())
            self.eaten += 1
            self.max_score = max(self.max_score, self.eaten)
            self.hunger -= self.HUNGER_LIMIT
            return self._render_state(), self.REWARD_EAT, False

        # if snake has made a normal step
        else:
            self.snake.shift_tail(self.ACTION_TO_DIRECTION[action])
            self.snake.set_position(new_snake_position)
            self.hunger += 1
            if self.hunger > self.HUNGER_LIMIT:
                return self._render_state(), self.REWARD_CRASH, True
            else:
                return self._render_state(), 0, False

    def render(self) -> None:
        if not self.visualize:
            return
        p = "="*self.SHAPE[1]*2+"\n"
        p += f"episode: {self.episode}\n"
        p += f"max score: {self.max_score}\n"
        p += f"score: {self.eaten} | steps: {self.steps}\n"
        for row in self._render_state():
            for cell in row:
                p += self.RENDER_MAP[cell]
            p += "\n"
        print(p)

    def close(self) -> None:
        pass
