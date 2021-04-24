import typing as T
from abc import ABC, abstractmethod
import random
import numpy as np
import asyncio
import torch

from ..agents import AnyAgent
from ..environments import Environment
from .individual import Individual
from .models import EvolutionProgress, EvolutionerParams, T_EParams, EvolvingFloat, EvolvingBool, EvolvingInt


class Evolutioner(ABC):
    def __init__(self,
                 env: Environment,
                 init_params: T_EParams,
                 evolutioner_params: EvolutionerParams):

        self.env: Environment = env
        self.evolutioner_params: EvolutionerParams = evolutioner_params
        self.init_params: T_EParams = init_params
        self.progress_callback: T.Callable[[EvolutionProgress], None] = lambda x: None
        self.fitness_function: T.Callable[[np.ndarray], float] = lambda x: np.mean(x).item()

    @abstractmethod
    def agent_factory(self, params: T_EParams, state_dict: T.Optional[dict]) -> AnyAgent:
        raise NotImplementedError()

    def set_progress_callback(self, cbk: T.Callable[[EvolutionProgress], None]) -> None:
        self.progress_callback = cbk

    def create_generation(self, params: T_EParams, state_dict: dict) -> T.Tuple[T.List[Individual], T.List[T_EParams]]:
        result: T.List[Individual] = []
        params_result: T.List[T_EParams] = []
        size = self.evolutioner_params.generation_size
        for i in range(size):
            mutated_params = self.mutate(params, i/size)
            agent = self.agent_factory(mutated_params, state_dict)
            result.append(Individual(agent, self.env))
            params_result.append(mutated_params)
        return result, params_result

    @staticmethod
    def mutate(params: T_EParams, scale_mutation: float) -> T_EParams:
        result = {}
        for k, v in params.items():
            if isinstance(v, EvolvingFloat):
                v_dict = v.__dict__
                mutation = (1 if random.random() > 0.5 else -1) * scale_mutation * v.mutation_step * random.random()
                v_dict["value"] += mutation
                result[k] = EvolvingFloat(**v_dict)
            elif isinstance(v, EvolvingInt):
                v_dict = v.__dict__
                mutation = (1 if random.random() > 0.5 else -1) * scale_mutation * v.mutation_step * random.random()
                v_dict["value"] = int(mutation + v_dict["value"])
                result[k] = EvolvingInt(**v_dict)
            elif isinstance(v, EvolvingBool):
                v_dict = v.__dict__
                mutation = scale_mutation * v.mutation_step
                if random.random() > mutation:
                    v_dict["value"] = not v_dict["value"]
                result[k] = EvolvingBool(**v_dict)
            else:
                raise ValueError(f"value from param {k} has type {type(v)}. Only types int, float and bool are valid")
        return result

    def evolve(self, finish_condition: T.Callable[[EvolutionProgress], bool]) -> None:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self._evolve(finish_condition))

    async def _evolve(self, finish_condition: T.Callable[[EvolutionProgress], bool]) -> None:
        torch.multiprocessing.set_start_method("spawn")
        params = self.init_params.copy()
        next_state_dict = None
        generation_count = 0
        while True:
            generation, mutated_params = self.create_generation(params, next_state_dict)
            results: T.List[T.Union[float, None]] = [None for _ in range(len(generation))]
            state_dicts: T.List[T.Union[dict, None]] = [None for _ in range(len(generation))]

            tasks: T.List[T.Coroutine] = []
            completed = []
            for i in range(len(generation)):
                async def task(index: int):
                    life_result, state_dict = await generation[index].parallel_life(index)
                    results[index] = self.fitness_function(life_result)
                    state_dicts[index] = state_dict
                    completed.append(None)
                    print("=============================")
                    print(f"completed {len(completed)}/{len(generation)}: {results[index]}")
                    for k, v in mutated_params[index].items():
                        print(k+":", v.value)

                tasks.append(task(i))
                if i >= len(generation) - 1:
                    await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
                    break
                if len(tasks) >= self.evolutioner_params.workers:
                    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = list(pending)

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
            next_state_dict = state_dicts[best_index]
