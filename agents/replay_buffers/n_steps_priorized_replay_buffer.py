from abc import ABC
import typing as T
import random

from .segment_trees import SumSegmentTree, MinSegmentTree
from .n_steps_replay_buffer import NStepsReplayBuffer
from .models import ReplayBufferEntry, NStepPrioritizedReplayBufferParams


class NStepsPrioritizedReplayBuffer(NStepsReplayBuffer, ABC):
    def __init__(self, rp: NStepPrioritizedReplayBufferParams):
        super(NStepsPrioritizedReplayBuffer, self).__init__(rp)
        if rp.alpha < 0:
            raise ValueError("alpha must be >= 0")

        self.max_priority: float = 1.0
        self.tree_ptr: int = 0
        self.rp: NStepPrioritizedReplayBufferParams = rp
        self.beta: float = rp.init_beta

        tree_capacity = 1
        while tree_capacity < self.rp.max_len:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def increase_beta(self):
        if self.beta < self.rp.final_beta:
            self.beta *= self.rp.increase_beta
        else:
            self.beta = self.rp.final_beta

    def add(self, entry: T.Union[ReplayBufferEntry]) -> bool:
        added = super(NStepsPrioritizedReplayBuffer, self).add(entry)
        if not added:
            return False
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.rp.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.rp.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.rp.max_len
        return True

    def sample(self, batch_size: int) -> T.List[ReplayBufferEntry]:
        indices = set(self._sample_proportional(batch_size))

        entries: T.List[ReplayBufferEntry] = [entry for i, entry in enumerate(self.records) if i in indices]
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
        p_total = self.sum_tree.sum(0, len(self) - 1)
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
