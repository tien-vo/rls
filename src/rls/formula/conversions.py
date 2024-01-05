__all__ = [
    "energy_to_speed",
    "speed_to_energy",
    "cartesian_to_FAC",
]

import numpy as np

from rls.data_types import SimQuantity


def energy_to_speed(energy, units, species="electron"):
    W0 = getattr(units, species).W0(units.c)
    return units.c * np.sqrt(1 - (1 + energy / W0) ** (-2))


def speed_to_energy(speed, units, species="electron"):
    W0 = getattr(units, species).W0(units.c)
    return W0 * (1 / np.sqrt(1 - (speed / units.c) ** 2) - 1)


def cartesian_to_FAC(g, ux, uy, uz, B0_x, B0_z, model, species="electron"):
    units = model.units
    species = getattr(units, species)
    V_factor = (units.L_factor / units.T_factor).to("1000 km/s")
    W_factor = units.W_factor

    B0_mag = np.sqrt(B0_x**2 + B0_z**2)
    u_mag = np.sqrt(ux**2 + uy**2 + uz**2)
    V_mag = u_mag / g
    V_para = (ux * B0_x + uz * B0_z) / B0_mag / g
    V_perp = np.sqrt(V_mag**2 - V_para**2)
    V_para = V_para * V_factor
    V_perp = V_perp * V_factor
    V_mag = V_mag * V_factor

    W = (g - 1) * W_factor.to("eV")
    A = np.arccos(V_para / V_mag)
    return V_para, V_perp, W, A
