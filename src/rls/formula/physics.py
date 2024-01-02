__all__ = ["lorentz_factor", "nb_lorentz_factor"]

import numba as nb
import numpy as np


def lorentz_factor(ux, uy, uz, c):
    return np.sqrt(1 + (ux**2 + uy**2 + uz**2) / c**2)


_jitter = nb.njit(
    [f"f8({'f8,' * 3}f8)", f"f8[::1]({'f8[::1],' * 3}f8)"], cache=True
)
nb_lorentz_factor = _jitter(lorentz_factor)
