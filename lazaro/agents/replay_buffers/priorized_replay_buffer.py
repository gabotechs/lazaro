import random
import typing as T
from abc import ABC

from .base.params import ReplayBufferEntry, PrioritizedReplayBufferParams
from .base.replay_buffer import ReplayBuffer
from ..base.models import LearningStep
from ..replay_buffers.base.segment_trees import SumSegmentTree, MinSegmentTree


class PrioritizedReplayBuffer(ReplayBuffer, ABC):
    def __init__(self, rp: PrioritizedReplayBufferParams = PrioritizedReplayBufferParams(), *args, **kwargs):
        if not isinstance(rp, PrioritizedReplayBufferParams):
            raise ValueError("argument rp must be an instance of PrioritizedReplayBufferParams")
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
        super(PrioritizedReplayBuffer, self).__init__(rp, *args, **kwargs)

    def _increase_beta(self, *_, **__):
        self.log.debug(f"increase beta for {type(self).__name__} triggered")
        if self.beta < self.rp.final_beta:
            self.beta += self.rp.increase_beta
        elif self.beta > self.rp.final_beta:
            self.beta = self.rp.final_beta

    def rp_add(self, entry: T.Union[ReplayBufferEntry]) -> bool:
        self.sum_tree[self.ptr] = self.max_priority ** self.rp.alpha
        self.min_tree[self.ptr] = self.max_priority ** self.rp.alpha
        return super(PrioritizedReplayBuffer, self).rp_add(entry)

    def rp_sample(self, batch_size: int) -> T.List[ReplayBufferEntry]:
        indices = self._sample_proportional(batch_size)

        entries: T.List[ReplayBufferEntry] = [self.records[i] for i in indices]
        for entry in entries:
            entry.weight = self._calculate_weight(entry.index)

        return entries

    def _refactor_priorities(self, indices: T.List[int], priorities: T.List[float]):
        if len(indices) != len(priorities):
            raise ValueError("indices and priorities must have the same length")

        for idx, priority in zip(indices, priorities):
            if priority <= 0:
                raise ValueError("priority for index "+str(idx)+" is less than 0")
            if not 0 <= idx < self.rp_get_length():
                raise ValueError("index "+str(idx)+" out of range")

            self.sum_tree[idx] = priority ** self.rp.alpha
            self.min_tree[idx] = priority ** self.rp.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, batch_size: int) -> T.List[int]:
        indices = []
        p_total = self.sum_tree.sum(0, self.rp_get_length())
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
        max_weight = (p_min * self.rp_get_length()) ** (-self.beta)

        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.rp_get_length()) ** (-self.beta)
        weight = weight / max_weight

        return weight

    def _update_priorities(self, learning_step: LearningStep) -> None:
        self.log.debug(f"update priorities for {type(self).__name__} triggered")
        self._refactor_priorities(
            [e.index for e in learning_step.batch],
            [abs(x - y) + 1e-7 for x, y in zip(learning_step.x, learning_step.y)]
        )

    def rp_clear(self):
        super(PrioritizedReplayBuffer, self).rp_clear()
        self.sum_tree.clear()
        self.min_tree.clear()

    def rp_link(self):
        self.log.info(f"linking {type(self).__name__} priority...")
        self.add_step_callback("prioritized replay buffer increase beta", self._increase_beta)
        self.add_learn_callback("prioritized replay buffer update priorities", self._update_priorities)

    def rp_get_stats(self) -> T.Dict[str, float]:
        stats = super(PrioritizedReplayBuffer, self).rp_get_stats()
        stats.update({"Prioritized Replay Buffer Beta": self.beta})
        return stats
