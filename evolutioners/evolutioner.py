import random
import numpy as np
import asyncio
import torch

from agents import Agent
from environments import Environment
from .individual import Individual
from .models import EvolutionProgress, EvolutionerParams
from .types import *


class Evolutioner:
    def __init__(self,
                 env: Environment,
                 evolutioner_params: EvolutionerParams,
                 init_params: T_EParams,
                 agent_factory: T.Callable[[T_EParams], Agent]):

        self.env: Environment = env
        self.evolutioner_params: EvolutionerParams = evolutioner_params
        self.init_params: T_EParams = init_params
        self.agent_factory: T.Callable[[T_EParams], Agent] = agent_factory
        self.progress_callback: T.Callable[[EvolutionProgress], None] = lambda x: None
        self.fitness_function: T.Callable[[np.ndarray], float] = lambda x: np.mean(x).item()

    def set_progress_callback(self, cbk: T.Callable[[EvolutionProgress], None]) -> None:
        self.progress_callback = cbk

    def create_generation(self, params: T_EParams) -> T.Tuple[T.List[Individual], T.List[T_EParams]]:
        result: T.List[Individual] = []
        params_result: T.List[T_EParams] = []
        size = self.evolutioner_params.generation_size
        for i in range(size):
            mutated_params = self.mutate(params, self.evolutioner_params.max_allowed_mutation*i/size)
            agent = self.agent_factory(mutated_params)
            result.append(Individual(agent, self.env))
            params_result.append(mutated_params)
        return result, params_result

    @staticmethod
    def mutate(params: T_EParams, max_mutation: float) -> T_EParams:
        def float_factor(x: float) -> float: return (max_mutation * 2 * random.random() + 1 - max_mutation) * x
        def int_factor(x: int) -> int: return int(float_factor(float(x)))
        def bool_factor(x: bool) -> bool: return x if random.random() > max_mutation else not x

        result = {}
        for k, v in params.items():
            if isinstance(v, float):
                result[k] = float_factor(v)
            elif isinstance(v, int):
                result[k] = int_factor(v)
            elif isinstance(v, bool):
                result[k] = bool_factor(v)
            else:
                raise ValueError(f"value from param {k} has type {type(v)}. Only types int, float and bool are valid")
        return result

    def evolve(self, finish_condition: T.Callable[[EvolutionProgress], bool]):
        torch.multiprocessing.set_start_method("spawn")
        params = self.init_params.copy()
        generation_count = 0
        while True:
            generation, mutated_params = self.create_generation(params)
            results: T.List[T.Union[float, None]] = [None for _ in range(len(generation))]

            loop = asyncio.new_event_loop()
            i = 0
            while i < len(generation):
                tasks: T.List[T.Coroutine] = []
                for _ in range(self.evolutioner_params.workers):
                    if i >= len(generation):
                        break

                    async def task(index: int):
                        life_result = await generation[index].parallel_life(index)
                        results[index] = self.fitness_function(life_result)

                    tasks.append(task(i))
                    i += 1

                loop.run_until_complete(asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED))

            best_index = np.argmax(results).item()
            progress = EvolutionProgress(
                results,
                mutated_params,
                generation_count,
                best_index
            )

            self.progress_callback(progress)
            if finish_condition(progress):
                break
            generation_count += 1
            params = mutated_params[best_index]
