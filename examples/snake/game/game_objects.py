import typing as T


class GameObject:
    def __init__(self):
        self.position: T.Optional[T.Tuple[int, int]] = None

    def set_position(self, position: T.Tuple[int, int]):
        self.position = position


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


class Apple(GameObject):
    APPLE = 3
