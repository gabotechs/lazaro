from .base.replay_buffer import ReplayBuffer


class LatestReplayBuffer(ReplayBuffer):
    def sample(self, limit: int = None):
        records = []
        ptr = self.ptr - 1
        while len(records) < limit:

            if ptr < 0:
                ptr = len(self.records) - 1
            record = self.records[ptr]
            if record is None:
                break
            records.insert(0, record)
            ptr -= 1
        return records
