__all__ = ["lorentz_factor", "nb_lorentz_factor", "f_maxwellian", "f_kappa"]

import numba as nb
import numpy as np
from astropy.units import Quantity
from scipy.special import gamma


def lorentz_factor(ux, uy, uz, c):
    return np.sqrt(1 + (ux**2 + uy**2 + uz**2) / c**2)


_jitter = nb.njit(
    [f"f8({'f8,' * 3}f8)", f"f8[::1]({'f8[::1],' * 3}f8)"], cache=True
)
nb_lorentz_factor = _jitter(lorentz_factor)


def f_maxwellian(V_para, V_perp, n, Vth_para, Vth_perp, V_drift):
    # Sanity checks
    for var in [V_para, V_perp, n, Vth_para, Vth_perp, V_drift]:
        assert isinstance(var, Quantity)

    A = (n / (np.pi**1.5 * Vth_perp**2 * Vth_para)).to("s3 km-3 cm-3")
    return A * np.exp(
        -(((V_para - V_drift) / Vth_para) ** 2 + (V_perp / Vth_perp) ** 2)
    )


def f_kappa(V_para, V_perp, n, Vth_para, Vth_perp, V_drift, kappa=3):
    # Sanity checks
    for var in [V_para, V_perp, n, Vth_para, Vth_perp, V_drift]:
        assert isinstance(var, Quantity)

    A = (
        n
        / ((np.pi * (kappa - 3 / 2)) ** (3 / 2) * Vth_perp**2 * Vth_para)
        * gamma(kappa + 1)
        / gamma(kappa - 1 / 2)
    )
    B = ((V_para - V_drift) / Vth_para) ** 2 + (V_perp / Vth_perp) ** 2
    return A.to("s3 km-3 cm-3") * (1 + B / (kappa - 3 / 2)) ** (-kappa - 1)
