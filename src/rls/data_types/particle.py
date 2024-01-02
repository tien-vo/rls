__all__ = ["Particle"]

from dataclasses import dataclass
from numbers import Number

import numpy as np

from rls.formula.physics import lorentz_factor

from .natural_units import NaturalUnits
from .species import Species


@dataclass
class Particle:
    species: Species
    t: float | np.ndarray
    g: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray
    state_components = ["g", "x", "y", "z", "ux", "uy", "uz"]

    def __post_init__(self):
        same_dimension = True
        _previous_component = None
        for component in self.state_components:
            _component = np.atleast_1d(getattr(self, component))
            setattr(self, component, _component)

            if _previous_component is None:
                _previous_component = _component
            else:
                same_dimension &= _component.ndim == _previous_component.ndim

        assert same_dimension, "State components must have the same dimension"

    def check_relativity(self, units: NaturalUnits):
        g = lorentz_factor(self.ux, self.uy, self.uz, units.light_speed.code)
        assert np.isclose(g, self.g).all(), "Relativity error!!!"

    def as_dict(self) -> dict:
        return {x: getattr(self, x) for x in self.state_components}

    def as_tuple(self) -> tuple:
        return tuple(getattr(self, x) for x in self.state_components)

    @property
    def is_initial_condition(self) -> bool:
        return isinstance(self.t, Number) and self.g.ndim == 1
