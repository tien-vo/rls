__all__ = ["WhistlerAtGradientModel"]

import numba as nb
import numpy as np

from .hyperbolic_gradient import HyperbolicGradientModel

background_field = HyperbolicGradientModel.field


@nb.njit(
    [
        f"UniTuple(f8, 2)({'f8,' * 4})",
        f"UniTuple(f8[::1], 2)(f8[::1], {'f8,' * 3})",
        f"UniTuple(f8[:, ::1], 2)(f8[:, ::1], {'f8,' * 3})",
    ],
    cache=True,
)
def dispersion_relation(wce, w_wce, wpe0, c):
    w = np.abs(w_wce * wce)
    k = (w / c) * np.sqrt(1 - wpe0**2 / (w * (w - wce)))
    return w, k


def resonant_velocity(wce, w_wce, wpe0, c):
    w, k = dispersion_relation(wce, w_wce, wpe0, c)
    V_res = (w - wce) / k
    return V_res


def resonant_width(V_perp, Bw, B0, wce, w_wce, wpe0, c):
    w, k = dispersion_relation(wce, w_wce, wpe0, c)
    N = c * k / w
    dV_para = (
        2
        * c
        * N
        / np.sqrt(N**2 - 1)
        * np.sqrt(np.abs((V_perp / c) * (Bw / B0) * wce / c / k))
    )
    return dV_para


@nb.njit(
    [
        f"f8[::1](f8, {'f8[::1],' * 2} f8)",
        f"f8[:, ::1](f8, {'f8[:, ::1],' * 2} f8)",
        f"f8[::1]({'f8[::1],' * 3} f8)",
        f"f8[:, ::1]({'f8[:, ::1],' * 3} f8)",
    ],
    cache=True,
)
def envelope(t, x, z, sw):
    return np.exp(-0.5 * (x**2 + z**2) / sw**2)


@nb.njit(
    [
        f"UniTuple(f8[::1], 6)(f8, {'f8[::1],' * 5} {'f8,' * 8})",
        f"UniTuple(f8[:, ::1], 6)(f8, {'f8[:, ::1],' * 5} {'f8,' * 8})",
        f"UniTuple(f8[::1], 6)({'f8[::1],' * 6} {'f8,' * 8})",
        f"UniTuple(f8[:, ::1], 6)({'f8[:, ::1],' * 6} {'f8,' * 8})",
    ],
    cache=True,
)
def wave_field(t, x, y, z, B0_x, B0_z, Bw, w_wce, theta, sw, wpe0, qe, me, c):
    B0_mag = np.sqrt(B0_x**2 + B0_z**2)
    wce = np.abs(qe * B0_mag / me)
    w, k = dispersion_relation(wce, w_wce, wpe0, c)
    kx = k * (B0_x * np.cos(theta) - B0_z * np.sin(theta)) / B0_mag
    kz = k * (B0_x * np.sin(theta) + B0_z * np.cos(theta)) / B0_mag
    phase = kx * x + kz * z - w * t

    A = envelope(t, x, z, sw)
    Ew_x = Bw * A * (w / k) * np.cos(phase) * (kz / k)
    Ew_y = -Bw * A * (w / k) * np.sin(phase)
    Ew_z = -Bw * A * (w / k) * np.cos(phase) * (kx / k)
    Bw_x = Bw * A * np.sin(phase) * (kz / k)
    Bw_y = Bw * A * np.cos(phase)
    Bw_z = -Bw * A * np.sin(phase) * (kx / k)
    return (Ew_x, Ew_y, Ew_z, Bw_x, Bw_y, Bw_z)


class WhistlerAtGradientModel(HyperbolicGradientModel):
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
        self.sw = sw * self.units.L_factor

    @property
    def field_args(self) -> tuple:
        Bw = self.Bw.code
        w_wce = self.w_wce
        theta = self.theta
        sw = self.sw.code
        R = self.R.code
        Bh = self.Bh.code
        B0 = self.B0.code
        wpe = self.units.electron.wp(self.n0, self.units.eps0).code
        q = self.units.electron.q.code
        m = self.units.electron.m.code
        c = self.units.c.code
        return (Bw, w_wce, theta, sw, R, Bh, B0, wpe, q, m, c)

    @property
    def background_field_args(self) -> tuple:
        return super().field_args

    @property
    def wave_field_args(self) -> tuple:
        Bw = self.Bw.code
        w_wce = self.w_wce
        theta = self.theta
        sw = self.sw.code
        wpe = self.units.electron.wp(self.n0, self.units.eps0).code
        q = self.units.electron.q.code
        m = self.units.electron.m.code
        c = self.units.c.code
        return (Bw, w_wce, theta, sw, wpe, q, m, c)

    @staticmethod
    @nb.njit(
        [
            f"UniTuple(f8[::1], 6)(f8, {'f8[::1],' * 3} {'f8,' * 11})",
            f"UniTuple(f8[:, ::1], 6)(f8, {'f8[:, ::1],' * 3} {'f8,' * 11})",
            f"UniTuple(f8[::1], 6)({'f8[::1],' * 4} {'f8,' * 11})",
            f"UniTuple(f8[:, ::1], 6)({'f8[:, ::1],' * 4} {'f8,' * 11})",
        ],
        cache=True,
    )
    def field(t, x, y, z, Bw, w_wce, theta, sw, R, Bh, B0, wpe0, qe, me, c):
        _, _, _, B0_x, _, B0_z = background_field(t, x, y, z, R, Bh, B0)
        Ew_x, Ew_y, Ew_z, Bw_x, Bw_y, Bw_z = wave_field(
            t, x, y, z, B0_x, B0_z, Bw, w_wce, theta, sw, wpe0, qe, me, c
        )
        return Ew_x, Ew_y, Ew_z, Bw_x + B0_x, Bw_y, Bw_z + B0_z
