__all__ = ["MagnetizedPlasmaUnits"]

from dataclasses import dataclass

import astropy.constants.si as constants
import numpy as np
from astropy.units import isclose

from rls.data_types import NaturalUnits, SimQuantity


@dataclass
class MagnetizedPlasmaUnits(NaturalUnits):
    magnetic_field: SimQuantity = SimQuantity(1.0, 1.0, "nT")

    def set_scale(self, scaling: str):
        self.scaling = scaling

        ms_c = self.scaling_species.mass.code.copy()
        for species in self.species:
            species.mass.code /= ms_c

    @property
    def vacuum_permittivity(self) -> SimQuantity:
        c_u = self.light_speed.user
        qs_u = np.abs(self.scaling_species.charge.user)
        ms_u = self.scaling_species.mass.user
        B0 = self.magnetic_field.user
        eps0_u = constants.eps0
        eps0_c = (eps0_u * ms_u**2 * c_u**3 / qs_u**3 / B0).decompose()
        return SimQuantity(eps0_c, eps0_u)

    @property
    def eps0(self) -> SimQuantity:
        return self.vacuum_permittivity

    @property
    def B_factor(self) -> SimQuantity:
        return self.magnetic_field.to("nT")

    @property
    def L_factor(self) -> SimQuantity:
        c = self.light_speed
        wcs = self.scaling_species.wc(self.B_factor)
        return (c / wcs).to("km")

    @property
    def T_factor(self) -> SimQuantity:
        wcs = self.scaling_species.wc(self.B_factor)
        return (1 / wcs).to("s")

    @property
    def Q_factor(self) -> SimQuantity:
        return np.abs(self.scaling_species.charge).to("C")

    @property
    def M_factor(self) -> SimQuantity:
        return self.scaling_species.mass.to("kg")

    @property
    def E_factor(self) -> SimQuantity:
        c = self.light_speed
        B0 = self.magnetic_field
        return (c * B0).to("mV/m")

    @property
    def W_factor(self) -> SimQuantity:
        return (self.M_factor * (self.L_factor / self.T_factor) ** 2).to("eV")

    def check_consistency(self):
        species = self.scaling_species
        qs_c = species.charge.code
        qs_u = species.charge.user
        ms_c = species.mass.code
        ms_u = species.mass.user
        c_u = self.light_speed.user
        W = self.W_factor.user

        super().check_consistency()
        assert isclose(W, ms_u * c_u**2)
        assert isclose(qs_c, np.sign(qs_u))
        assert isclose(ms_c, 1.0)
