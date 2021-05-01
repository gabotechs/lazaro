import typing as T


class GameObject:
    def __init__(self):
        self.position: T.Optional[T.Tuple[int, int]] = None

    def set_position(self, position: T.Tuple[int, int]):
        self.position = position
