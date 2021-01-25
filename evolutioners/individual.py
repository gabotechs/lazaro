import typing as T
import multiprocessing
import asyncio
import os
import json
import time
from agents import AnyAgent, TrainingProgress
from environments import Environment


class Individual:
    def __init__(self, agent: AnyAgent, env: Environment):
        self.agent: AnyAgent = agent
        self.env: Environment = env
        self.history: T.List[float] = []

    def _on_progress(self, progress: TrainingProgress) -> bool:
        self.history.append(progress.total_reward)
        return False

    def life(self, dump_file: str = "") -> T.List[float]:
        self.agent.add_progress_callback(self._on_progress)
        self.agent.tensor_board_log = False
        self.agent.save_progress = False
        self.agent.train(self.env)
        self.env.close()
        if dump_file != "":
            json.dump(self.history, open(dump_file, "w"))
        return self.history

    async def parallel_life(self, worker_id: int, timeout: int = 0) -> T.List[float]:
        file_name = str(worker_id) + "_" + str(time.time()) + ".json"

        process = multiprocessing.Process(target=self.life, args=(file_name,))
        process.start()
        start = time.time()
        while True:
            if timeout and time.time()-start > timeout:
                process.kill()
                raise TimeoutError("worker "+str(worker_id)+" timeout")

            if os.path.isfile(file_name):
                await asyncio.sleep(1)
                result = json.load(open(file_name))
                os.remove(file_name)
                return result
            elif not process.is_alive():
                raise RuntimeError("worker "+str(worker_id)+" died without returning a result")

            await asyncio.sleep(1)
