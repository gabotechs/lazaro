from abc import ABC, abstractmethod
import typing as T

T_entry = T.TypeVar("T_entry")


class ReplayBufferInterface(T.Generic[T_entry], ABC):
    @abstractmethod
    def rp_get_length(self):
        raise NotImplementedError()

    @abstractmethod
    def rp_clear(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def rp_add(self, entry: T_entry) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def rp_sample(self, limit: int) -> T.List[T_entry]:
        raise NotImplementedError()

    @abstractmethod
    def rp_link(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def rp_get_stats(self) -> T.Dict[str, float]:
        raise NotImplementedError()
