__all__ = ["HyperbolicGradientModel"]

import numba as nb
import numpy as np

from .uniform_plasma import UniformPlasmaModel

jitter = nb.njit(
    [
        "f8(f8, f8, f8, f8)",
        "f8[::1](f8[::1], f8, f8, f8)",
        "f8[:, ::1](f8[:, ::1], f8, f8, f8)",
    ],
    cache=True,
)
eta = jitter(lambda z, R, Bh, B0: 1 - 0.5 * (Bh / B0) * (1 + np.tanh(z / R)))
d_eta = jitter(lambda z, R, Bh, B0: -0.5 * (Bh / B0 / R) / np.cosh(z / R) ** 2)


class HyperbolicGradientModel(UniformPlasmaModel):
    eta = staticmethod(eta)
    d_eta = staticmethod(d_eta)

    units = UniformPlasmaModel.units
    units.set_scale("electron")

    def __init__(
        self,
        R: float = 3.5,
        Bh: float = 0.5,
        B0: float = 1.0,
        wpe_wce: float = 100.0,
    ):
        super().__init__(E0=0.0, B0=B0, wpe_wce=wpe_wce)
        self.R = R * self.units.L_factor
        self.Bh = Bh * self.units.B_factor
        self.check_inhomogeneity_consistency()

    def check_inhomogeneity_consistency(self):
        de0 = self.units.electron.cwp(
            self.density,
            self.units.vacuum_permittivity,
            self.units.light_speed,
        )
        R_de0 = self.R / de0
        assert np.isclose(R_de0.code, R_de0.user.value)

    @property
    def field_args(self) -> tuple:
        return (self.R.code, self.Bh.code, self.B0.code)

    @staticmethod
    @nb.njit(
        [
            f"UniTuple(f8, 6)(f8, {'f8,' * 3} {'f8,' * 3})",
            f"UniTuple(f8[::1], 6)(f8, {'f8[::1],' * 3} {'f8,' * 3})",
            f"UniTuple(f8[:, ::1], 6)(f8, {'f8[:, ::1],' * 3} {'f8,' * 3})",
            f"UniTuple(f8[::1], 6)({'f8[::1],' * 4} {'f8,' * 3})",
            f"UniTuple(f8[:, ::1], 6)({'f8[:, ::1],' * 4} {'f8,' * 3})",
        ],
        cache=True,
    )
    def field(t, x, y, z, R, Bh, B0) -> tuple:
        Bx = -x * B0 * d_eta(z, R, Bh, B0)
        Bz = B0 * eta(z, R, Bh, B0)
        if isinstance(x, np.float64):
            Ex, Ey, Ez, By = 0.0, 0.0, 0.0, 0.0
        else:
            Ex = np.zeros_like(x)
            Ey = np.zeros_like(y)
            Ez = np.zeros_like(z)
            By = np.zeros_like(y)
        return (Ex, Ey, Ez, Bx, By, Bz)
