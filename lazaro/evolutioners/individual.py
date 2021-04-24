import asyncio
import json
import multiprocessing
import os
import pickle
import time
import typing as T

from ..agents import AnyAgent, TrainingProgress
from ..environments import Environment


class Individual:
    def __init__(self, agent: AnyAgent, env: Environment):
        self.agent: AnyAgent = agent
        self.env: Environment = env
        self.history: T.List[float] = []

    def _on_progress(self, progress: TrainingProgress) -> bool:
        self.history.append(progress.total_reward)
        return False

    def life(self, dump_file: str = "", state_dump_file: str = "") -> T.List[float]:
        self.agent.add_progress_callback("evolution callback", self._on_progress)
        self.agent.tensor_board_log = False
        self.agent.save_progress = False
        self.agent.train(self.env)
        self.env.close()
        if dump_file != "":
            json.dump(self.history, open(dump_file, "w"))
        if state_dump_file != "":
            pickle.dump(self.agent.get_state_dict(), open(state_dump_file, "wb"))
        return self.history

    async def parallel_life(self, worker_id: int, timeout: int = 0) -> T.Tuple[T.List[float], dict]:
        now = str(time.time())
        file_name = f"{worker_id}_result_{now}.json"
        state_file_name = f"{worker_id}_state_dict_{now}.json"

        process = multiprocessing.Process(target=self.life, args=(file_name, state_file_name))
        process.start()
        start = time.time()
        while True:
            if timeout and time.time()-start > timeout:
                process.kill()
                raise TimeoutError("worker "+str(worker_id)+" timeout")

            if os.path.isfile(file_name) and os.path.isfile(state_file_name):
                await asyncio.sleep(1)
                result = json.load(open(file_name))
                state_dict = pickle.load(open(state_file_name, "rb"))
                os.remove(file_name)
                os.remove(state_file_name)
                return result, state_dict
            elif not process.is_alive():
                raise RuntimeError("worker "+str(worker_id)+" died without returning a result")

            await asyncio.sleep(1)
