import typing as T
import random

from agents.replay_buffers.base.segment_trees import SumSegmentTree, MinSegmentTree
from .base.replay_buffer import ReplayBuffer
from .base.models import ReplayBufferEntry, PrioritizedReplayBufferParams


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, rp: PrioritizedReplayBufferParams = PrioritizedReplayBufferParams()):
        super(PrioritizedReplayBuffer, self).__init__(rp)
        if rp.alpha < 0:
            raise ValueError("alpha must be >= 0")

        self.max_priority: float = 1.0
        self.rp: PrioritizedReplayBufferParams = rp
        self.beta: float = rp.init_beta

        tree_capacity = 1
        while tree_capacity < self.rp.max_len:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def increase_beta(self):
        if self.beta < self.rp.final_beta:
            self.beta += self.rp.increase_beta
        elif self.beta > self.rp.final_beta:
            self.beta = self.rp.final_beta

    def add(self, entry: T.Union[ReplayBufferEntry]) -> bool:
        self.sum_tree[self.ptr] = self.max_priority ** self.rp.alpha
        self.min_tree[self.ptr] = self.max_priority ** self.rp.alpha
        return super(PrioritizedReplayBuffer, self).add(entry)

    def sample(self, batch_size: int) -> T.List[ReplayBufferEntry]:
        indices = self._sample_proportional(batch_size)

        entries: T.List[ReplayBufferEntry] = [self.records[i] for i in indices]
        for entry in entries:
            entry.weight = self._calculate_weight(entry.index)

        return entries

    def update_priorities(self, indices: T.List[int], priorities: T.List[float]):
        if len(indices) != len(priorities):
            raise ValueError("indices and priorities must have the same length")

        for idx, priority in zip(indices, priorities):
            if priority <= 0:
                raise ValueError("priority for index "+str(idx)+" is less than 0")
            if not 0 <= idx < len(self):
                raise ValueError("index "+str(idx)+" out of range")

            self.sum_tree[idx] = priority ** self.rp.alpha
            self.min_tree[idx] = priority ** self.rp.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, batch_size: int) -> T.List[int]:
        indices = []
        p_total = self.sum_tree.sum(0, len(self))
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int):
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-self.beta)

        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-self.beta)
        weight = weight / max_weight

        return weight

    def clear(self):
        super(PrioritizedReplayBuffer, self).clear()
        self.sum_tree.clear()
        self.min_tree.clear()
