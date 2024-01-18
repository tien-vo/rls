__all__ = ["WhistlerInMagneticHoleModel"]

import numba as nb
import numpy as np

from ..magnetic_hole import MagneticHoleModel
from .properties import (
    dispersion_relation,
    envelope,
    resonant_velocity,
    resonant_width,
    wave_field,
)

background_field = MagneticHoleModel.field


class WhistlerInMagneticHoleModel(MagneticHoleModel):
    envelope = staticmethod(envelope)
    dispersion_relation = staticmethod(dispersion_relation)
    background_field = staticmethod(background_field)
    wave_field = staticmethod(wave_field)
    resonant_velocity = staticmethod(resonant_velocity)
    resonant_width = staticmethod(resonant_width)

    def __init__(
        self,
        Bw: float = 0.01,
        w_wce: float = 0.1,
        theta: float = 0.0,
        xw: float = 0.0,
        zw: float = -3.5,
        sw: float = 3.5,
        R: float = 3.5,
        Bh: float = 0.5,
        B0: float = 1.0,
        wpe_wce: float = 100.0,
    ):
        super().__init__(R, Bh, B0, wpe_wce)
        self.Bw = np.abs(Bw * self.units.B_factor)
        self.w_wce = w_wce
        self.theta = theta
        self.xw = xw * self.units.L_factor
        self.zw = zw * self.units.L_factor
        self.sw = sw * self.units.L_factor

    @property
    def field_args(self) -> tuple:
        Bw = self.Bw.code
        w_wce = self.w_wce
        theta = self.theta
        xw = self.xw.code
        zw = self.zw.code
        sw = self.sw.code
        R = self.R.code
        Bh = self.Bh.code
        B0 = self.B0.code
        wpe = self.units.electron.wp(self.n0, self.units.eps0).code
        q = self.units.electron.q.code
        m = self.units.electron.m.code
        c = self.units.c.code
        return (Bw, w_wce, theta, xw, zw, sw, R, Bh, B0, wpe, q, m, c)

    @property
    def background_field_args(self) -> tuple:
        return super().field_args

    @property
    def wave_field_args(self) -> tuple:
        Bw = self.Bw.code
        w_wce = self.w_wce
        theta = self.theta
        xw = self.xw.code
        zw = self.zw.code
        sw = self.sw.code
        wpe = self.units.electron.wp(self.n0, self.units.eps0).code
        q = self.units.electron.q.code
        m = self.units.electron.m.code
        c = self.units.c.code
        return (Bw, w_wce, theta, xw, zw, sw, wpe, q, m, c)

    @staticmethod
    @nb.njit(
        [
            f"UniTuple(f8[::1], 7)(f8, {'f8[::1],' * 3} {'f8,' * 13})",
            f"UniTuple(f8[:, ::1], 7)(f8, {'f8[:, ::1],' * 3} {'f8,' * 13})",
            f"UniTuple(f8[::1], 7)({'f8[::1],' * 4} {'f8,' * 13})",
            f"UniTuple(f8[:, ::1], 7)({'f8[:, ::1],' * 4} {'f8,' * 13})",
        ],
        cache=True,
    )
    def field(
        t, x, Iw, z, Bw, w_wce, theta, xw, zw, sw, R, Bh, B0, wpe0, qe, me, c
    ):
        args_1 = (R, Bh, B0)
        args_2 = (Bw, w_wce, theta, xw, zw, sw, wpe0, qe, me, c)
        _, _, _, B0_x, _, B0_z = background_field(t, x, Iw, z, *args_1)
        Ew_x, Ew_y, Ew_z, Bw_x, Bw_y, Bw_z = wave_field(
            t, x, Iw, z, B0_x, B0_z, *args_2
        )

        if x.ndim == 1:
            for i in range(x.size):
                if Iw[i] > 1.0:
                    Ew_x[i] = 0.0
                    Ew_y[i] = 0.0
                    Ew_z[i] = 0.0
                    Bw_x[i] = 0.0
                    Bw_y[i] = 0.0
                    Bw_z[i] = 0.0

        _Iw = envelope(t, x, z, xw, zw, sw) / (sw * np.sqrt(2 * np.pi))
        return _Iw, Ew_x, Ew_y, Ew_z, Bw_x + B0_x, Bw_y, Bw_z + B0_z
