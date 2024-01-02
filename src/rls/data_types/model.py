__all__ = ["Model"]

from abc import ABC, abstractmethod, abstractproperty


class Model(ABC):
    @abstractproperty
    def field_args(self) -> tuple:
        raise NotImplementedError()

    @abstractmethod
    def field(t, x, y, z, *field_args) -> tuple:
        raise NotImplementedError()
