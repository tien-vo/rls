__all__ = ["UniformPlasmaModel"]

import numba as nb
import numpy as np

from rls.data_types import Model, Particle, SimQuantity
from rls.units import MagnetizedPlasmaUnits


class UniformPlasmaModel(Model):
    units = MagnetizedPlasmaUnits()

    def __init__(
        self, E0: float = 0.0, B0: float = 1.0, wpe_wce: float = 100.0
    ):
        self.E0 = E0 * self.units.E_factor
        self.B0 = B0 * self.units.B_factor
        self.wpe_wce = wpe_wce

        self.check_density_consistency()

    def check_density_consistency(self):
        units = self.units
        wpe = units.electron.wp(self.density, units.vacuum_permittivity)
        wce = units.electron.wc(self.B0)
        wpe_wce = wpe / wce
        assert np.isclose(self.wpe_wce, wpe_wce.code)
        assert np.isclose(self.wpe_wce, wpe_wce.user.value)

    @property
    def density(self) -> SimQuantity:
        eps0 = self.units.vacuum_permittivity
        qe = self.units.electron.charge
        me = self.units.electron.mass
        wce = self.units.electron.wc(self.B0)
        wpe = self.wpe_wce * wce
        return (wpe**2 * eps0 * me / qe**2).to("cm-3")

    @property
    def n0(self) -> SimQuantity:
        return self.density

    def analytical_solution(self, t: np.ndarray, initial_conditions: Particle):
        t = t[:, np.newaxis]
        g0 = initial_conditions.g[np.newaxis, :]
        x0 = initial_conditions.x[np.newaxis, :]
        y0 = initial_conditions.y[np.newaxis, :]
        z0 = initial_conditions.z[np.newaxis, :]
        ux0 = initial_conditions.ux[np.newaxis, :]
        uy0 = initial_conditions.uy[np.newaxis, :]
        uz0 = initial_conditions.uz[np.newaxis, :]

        species = initial_conditions.species
        c = self.units.light_speed.code
        Omega = species.Omega(self.B0).code
        V_drift = species.V_ExB(self.E0, self.B0).code
        vperp = np.sqrt(ux0**2 + uy0**2) / g0
        delta = -np.arctan2(V_drift + uy0 / g0, ux0 / g0)

        phase = Omega * t + delta
        x = x0 + vperp / Omega * (np.sin(phase) - np.sin(delta))
        y = y0 - V_drift * t + vperp / Omega * (np.cos(phase) - np.cos(delta))
        z = z0 + (uz0 / g0) * t
        vx = vperp * np.cos(phase)
        vy = -V_drift - vperp * np.sin(phase)
        vz = uz0 / g0 * np.ones_like(t)
        g = 1 / np.sqrt(1 - (vx**2 + vy**2 + vz**2) / c**2)

        solutions = Particle(species, t, g, x, y, z, g * vx, g * vy, g * vz)
        solutions.check_relativity(self.units)
        return solutions

    @property
    def field_args(self) -> tuple[float]:
        return (self.E0.code, self.B0.code)

    @staticmethod
    @nb.njit(
        [
            f"UniTuple(f8[::1], 6)(f8, {'f8[::1],' * 3} {'f8,' * 2})",
        ],
        cache=True,
    )
    def field(t, x, y, z, E0, B0):
        Ex = np.zeros_like(x) + E0
        Ey = np.zeros_like(y)
        Ez = np.zeros_like(z)
        Bx = np.zeros_like(x)
        By = np.zeros_like(y)
        Bz = np.zeros_like(z) + B0
        return (Ex, Ey, Ez, Bx, By, Bz)
