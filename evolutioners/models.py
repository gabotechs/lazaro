from .types import *


class EvolutionerParams:
    def __init__(self, workers: int, generation_size: int, max_allowed_mutation: float):
        self.workers: int = workers
        self.generation_size: int = generation_size
        self.max_allowed_mutation: float = max_allowed_mutation


class EvolutionProgress:
    def __init__(self, results: T.List[float], params: T.List[T_EParams], generation: int, best_index: int):
        self.best_index: int = best_index
        self.results: T.List[float] = results
        self.params: T.List[T_EParams] = params
        self.generation: int = generation
