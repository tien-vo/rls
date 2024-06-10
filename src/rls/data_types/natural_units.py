__all__ = ["NaturalUnits"]

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import astropy.constants.si as constants
from astropy.units import isclose

from .quantity import SimQuantity
from .species import Electron, Ion, Species


@dataclass
class NaturalUnits(ABC):
    scaling: str = "electron"
    light_speed: SimQuantity = SimQuantity(1.0, constants.c)

    _species: list[str] = []

    def __post_init__(self):
        self.add_species(Ion)
        self.add_species(Electron)

    def add_species(self, species: Species):
        if species.name not in self._species:
            self._species.append(species.name)

        setattr(self, species.name, species)

    @property
    def species(self) -> Iterator[Species]:
        for species in self._species:
            yield getattr(self, species)

    @property
    def scaling_species(self):
        return getattr(self, self.scaling)

    @abstractmethod
    def L_factor(self) -> SimQuantity:
        raise NotImplementedError()

    @abstractmethod
    def T_factor(self) -> SimQuantity:
        raise NotImplementedError()

    @abstractmethod
    def M_factor(self) -> SimQuantity:
        raise NotImplementedError()

    @abstractmethod
    def Q_factor(self) -> SimQuantity:
        raise NotImplementedError()

    @abstractmethod
    def E_factor(self) -> SimQuantity:
        raise NotImplementedError()

    @abstractmethod
    def B_factor(self) -> SimQuantity:
        raise NotImplementedError()

    def check_consistency(self):
        species = self.scaling_species
        c_c = self.light_speed.code
        c_u = self.light_speed.user
        eps0_c = self.vacuum_permittivity.code
        eps0_u = self.vacuum_permittivity.user
        r_c = species.charge.code / species.mass.code
        r_u = species.charge.user / species.mass.user

        L = self.L_factor.user
        T = self.T_factor.user
        Q = self.Q_factor.user
        M = self.M_factor.user
        E = self.E_factor.user
        B = self.B_factor.user
        assert isclose(T, L * c_c / c_u)
        assert isclose(
            Q, L * (c_u / c_c) ** 2 * (eps0_u / eps0_c) * (r_c / r_u)
        )
        assert isclose(
            M, L * (c_u / c_c) ** 2 * (eps0_u / eps0_c) * (r_c / r_u) ** 2
        )
        assert isclose(E, M * L / Q / T**2)
        assert isclose(B, E * T / L)

    @property
    def c(self) -> SimQuantity:
        return self.light_speed
