__all__ = ["push_r", "push_u"]

import numba as nb
import numpy as np

from rls.formula.physics import nb_lorentz_factor


@nb.njit("f8[::1](f8[::1], f8[::1], f8[::1], f8)", cache=True)
def push_r(r_i, gamma, u_push, dt):
    return r_i + dt * u_push / gamma


@nb.njit(
    f"UniTuple(f8[::1], 4)({'f8[::1],' * 9}{'f8,' * 4})",
    cache=True,
)
def push_u(ux_i, uy_i, uz_i, Ex, Ey, Ez, Bx, By, Bz, qs, ms, c, dt):
    qdt_2m = qs * dt / 2 / ms

    # ---- First half-acceleration (Kick)
    ux_m = ux_i + qdt_2m * Ex
    uy_m = uy_i + qdt_2m * Ey
    uz_m = uz_i + qdt_2m * Ez
    g_m = nb_lorentz_factor(ux_m, uy_m, uz_m, c)

    # ---- Gyration (Drift)
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    tan_theta_2 = np.tan(qdt_2m * B_mag / g_m)
    tx = tan_theta_2 * Bx / B_mag
    ty = tan_theta_2 * By / B_mag
    tz = tan_theta_2 * Bz / B_mag
    s = 2 / (1 + tx**2 + ty**2 + tz**2)
    ux_p = ux_m + s * (
        -(ty**2) * ux_m
        + ty * (tx * uy_m - uz_m)
        + tz * (-tz * ux_m + uy_m + tx * uz_m)
    )
    uy_p = uy_m + s * (
        -(tx**2) * uy_m
        + tx * (ty * ux_m + uz_m)
        - tz * (ux_m + tz * uy_m - ty * uz_m)
    )
    uz_p = uz_m + s * (
        ty * (ux_m + tz * uy_m)
        - ty**2 * uz_m
        - tx * (-tz * ux_m + uy_m + tx * uz_m)
    )

    # ---- Second half-acceleration (Kick)
    ux_f = ux_p + qdt_2m * Ex
    uy_f = uy_p + qdt_2m * Ey
    uz_f = uz_p + qdt_2m * Ez
    g_f = nb_lorentz_factor(ux_f, uy_f, uz_f, c)

    return (g_f, ux_f, uy_f, uz_f)
