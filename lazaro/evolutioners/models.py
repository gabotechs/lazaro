import typing as T


class EvolvingInt:
    def __init__(self, value: int, min_limit: int, max_limit: int, mutation_step: float):
        if value > max_limit:
            value = max_limit
        elif value < min_limit:
            value = min_limit
        self.value: int = value
        self.min_limit: int = min_limit
        self.max_limit: int = max_limit
        self.mutation_step: float = mutation_step


class EvolvingFloat:
    def __init__(self, value: float, min_limit: float, max_limit: float, mutation_step: float):
        if value > max_limit:
            value = max_limit
        elif value < min_limit:
            value = min_limit
        self.value: float = value
        self.min_limit: float = min_limit
        self.max_limit: float = max_limit
        self.mutation_step: float = mutation_step


class EvolvingBool:
    def __init__(self, value: bool, mutation_step: float):
        self.value: float = value
        self.mutation_step: float = mutation_step


class EvolutionerParams:
    def __init__(self, workers: int, generation_size: int):
        self.workers: int = workers
        self.generation_size: int = generation_size


T_EParams = T.Dict[str, T.Union[EvolvingInt, EvolvingBool, EvolvingFloat]]


class EvolutionProgress:
    def __init__(self, results: T.List[float], params: T.List[T_EParams], generation: int, best_index: int):
        self.best_index: int = best_index
        self.results: T.List[float] = results
        self.params: T.List[T_EParams] = params
        self.generation: int = generation
