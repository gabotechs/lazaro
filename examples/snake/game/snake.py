from .game_object import GameObject
import typing as T


class Snake(GameObject):
    HEAD = 1
    TAIL = 2

    def __init__(self):
        super(Snake, self).__init__()
        self.tail = []

    def shift_tail(self, direction: T.Tuple[int, int]):
        self.tail.insert(0, direction)
        self.tail.pop()

    def grow_tail(self, direction: T.Tuple[int, int]):
        self.tail.insert(0, direction)

    def clear_tail(self):
        self.tail.clear()
