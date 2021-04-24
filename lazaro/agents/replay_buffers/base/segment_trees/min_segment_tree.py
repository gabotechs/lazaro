from .segment_tree import SegmentTree


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super(MinSegmentTree, self).__init__(capacity=capacity, operation=min, init_value=float("inf"))

    def min(self, start: int = 0, end: int = 0) -> float:
        return super(MinSegmentTree, self).operate(start, end)
