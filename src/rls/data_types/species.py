__all__ = ["Species", "Ion", "Electron"]

from dataclasses import dataclass

import astropy.constants.si as constants
import numpy as np

from .quantity import SimQuantity


@dataclass
class Species:
    name: str
    charge: SimQuantity
    mass: SimQuantity

    @property
    def q(self) -> SimQuantity:
        return self.charge

    @property
    def m(self) -> SimQuantity:
        return self.mass

    def signed_angular_cyclotron_frequency(
        self, magnetic_field: SimQuantity
    ) -> SimQuantity:
        Omega = self.q * magnetic_field / self.m
        return Omega.to("Hz")

    def Omega(self, magnetic_field: SimQuantity) -> SimQuantity:
        return self.signed_angular_cyclotron_frequency(magnetic_field)

    def unsigned_angular_cyclotron_frequency(
        self, magnetic_field: SimQuantity
    ) -> SimQuantity:
        return np.abs(self.Omega(magnetic_field))

    def wc(self, magnetic_field: SimQuantity):
        return self.unsigned_angular_cyclotron_frequency(magnetic_field)

    def cyclotron_period(self, magnetic_field: SimQuantity) -> SimQuantity:
        Tc = 2 * np.pi / self.wc(magnetic_field)
        return Tc.to("s")

    def Tc(self, magnetic_field: SimQuantity) -> SimQuantity:
        return self.cyclotron_period(magnetic_field)

    def angular_plasma_frequency(
        self, density: SimQuantity, vacuum_permittivity: SimQuantity
    ) -> SimQuantity:
        wp = np.sqrt(density * self.q**2 / vacuum_permittivity / self.m)
        return wp.to("Hz")

    def wp(self, density: SimQuantity, vacuum_permittivity: SimQuantity):
        return self.angular_plasma_frequency(density, vacuum_permittivity)

    def inertial_length(
        self,
        density: SimQuantity,
        vacuum_permittivity: SimQuantity,
        light_speed: SimQuantity,
    ) -> SimQuantity:
        cwp = light_speed / self.wp(density, vacuum_permittivity)
        return cwp.to("km")

    def cwp(
        self,
        density: SimQuantity,
        vacuum_permittivity: SimQuantity,
        light_speed: SimQuantity,
    ) -> SimQuantity:
        return self.inertial_length(density, vacuum_permittivity, light_speed)

    def rest_energy(self, light_speed: SimQuantity) -> SimQuantity:
        W0 = self.m * light_speed**2
        return W0.to("eV")

    def W0(self, light_speed: SimQuantity) -> SimQuantity:
        return self.rest_energy(light_speed)

    def drift_velocity_ExB(
        self, electric_field: SimQuantity, magnetic_field: SimQuantity
    ) -> SimQuantity:
        V_ExB = electric_field / magnetic_field
        return V_ExB.to("km/s")

    def V_ExB(
        self, electric_field: SimQuantity, magnetic_field: SimQuantity
    ) -> SimQuantity:
        return self.drift_velocity_ExB(electric_field, magnetic_field)


Electron = Species(
    "electron",
    charge=SimQuantity(-1.0, -constants.e),
    mass=SimQuantity(1.0, constants.m_e),
)
Ion = Species(
    "ion",
    charge=SimQuantity(1.0, constants.e),
    mass=SimQuantity((constants.m_p / constants.m_e).value, constants.m_p),
)
