import typing as T
import random
import numpy as np
import asyncio
import torch

from agents import Agent
from environments import Environment
from .individual import Individual
from .models import EvolutionProgress, EvolutionerParams, T_EParams, EvolvingFloat, EvolvingBool, EvolvingInt


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
            mutated_params = self.mutate(params, i/size)
            agent = self.agent_factory(mutated_params)
            result.append(Individual(agent, self.env))
            params_result.append(mutated_params)
        return result, params_result

    @staticmethod
    def mutate(params: T_EParams, scale_mutation: float) -> T_EParams:
        result = {}
        for k, v in params.items():
            if isinstance(v, EvolvingFloat):
                v_dict = v.__dict__
                mutation = 1 + scale_mutation * v.mutation_factor * random.random()
                v_dict["value"] = mutation * v.value if random.random() > 0.5 else v.value / mutation
                result[k] = EvolvingFloat(**v_dict)
            elif isinstance(v, EvolvingInt):
                v_dict = v.__dict__
                mutation = 1 + scale_mutation * v.mutation_factor * random.random()
                v_dict["value"] = int(mutation * v.value if random.random() > 0.5 else v.value / mutation)
                result[k] = EvolvingInt(**v_dict)
            elif isinstance(v, EvolvingBool):
                v_dict = v.__dict__
                mutation = scale_mutation * v.mutation_factor
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
        generation_count = 0
        while True:
            generation, mutated_params = self.create_generation(params)
            results: T.List[T.Union[float, None]] = [None for _ in range(len(generation))]

            tasks: T.List[T.Coroutine] = []
            completed = []
            for i in range(len(generation)):
                async def task(index: int):
                    life_result = await generation[index].parallel_life(index)
                    results[index] = self.fitness_function(life_result)
                    completed.append(None)
                    print(f"completed {len(completed)}/{len(generation)}")

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
