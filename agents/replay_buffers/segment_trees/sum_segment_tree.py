import operator

from .segment_tree import SegmentTree


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super(SumSegmentTree, self).__init__(capacity=capacity, operation=operator.add, init_value=0.0)

    def sum(self, start: int = 0, end: int = 0) -> float:
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


