__all__ = [
    "envelope",
    "dispersion_relation",
    "wave_field",
    "resonant_velocity",
    "resonant_width",
]

import numba as nb
import numpy as np


@nb.njit(
    [
        f"f8[::1](f8, {'f8[::1],' * 2} {'f8,' * 3})",
        f"f8[:, ::1](f8, {'f8[:, ::1],' * 2} {'f8,' * 3})",
        f"f8[::1]({'f8[::1],' * 3} {'f8,' * 3})",
        f"f8[:, ::1]({'f8[:, ::1],' * 3} {'f8,' * 3})",
    ],
    cache=True,
)
def envelope(t, x, z, xw, zw, sw):
    return np.exp(-0.5 * ((x - xw) ** 2 + (z - zw) ** 2) / sw**2)


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


@nb.njit(
    [
        f"UniTuple(f8[::1], 6)(f8, {'f8[::1],' * 5} {'f8,' * 10})",
        f"UniTuple(f8[:, ::1], 6)(f8, {'f8[:, ::1],' * 5} {'f8,' * 10})",
        f"UniTuple(f8[::1], 6)({'f8[::1],' * 6} {'f8,' * 10})",
        f"UniTuple(f8[:, ::1], 6)({'f8[:, ::1],' * 6} {'f8,' * 10})",
    ],
    cache=True,
)
def wave_field(
    t, x, y, z, B0_x, B0_z, Bw, w_wce, theta, xw, zw, sw, wpe0, qe, me, c
):
    B0_mag = np.sqrt(B0_x**2 + B0_z**2)
    wce = np.abs(qe * B0_mag / me)
    w, k = dispersion_relation(wce, w_wce, wpe0, c)
    kx = k * (B0_x * np.cos(theta) - B0_z * np.sin(theta)) / B0_mag
    kz = k * (B0_x * np.sin(theta) + B0_z * np.cos(theta)) / B0_mag
    phase = kx * x + kz * z - w * t

    A = envelope(t, x, z, xw, zw, sw)
    Ew_x = Bw * A * (w / k) * np.cos(phase) * (kz / k)
    Ew_y = -Bw * A * (w / k) * np.sin(phase)
    Ew_z = -Bw * A * (w / k) * np.cos(phase) * (kx / k)
    Bw_x = Bw * A * np.sin(phase) * (kz / k)
    Bw_y = Bw * A * np.cos(phase)
    Bw_z = -Bw * A * np.sin(phase) * (kx / k)
    return (Ew_x, Ew_y, Ew_z, Bw_x, Bw_y, Bw_z)


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
