import colorcet
import config as cf
import matplotlib as mpl
import numpy as np
from rls.formula.conversions import cartesian_to_FAC
from tvolib import mpl_utils as mu

cf.sim.name = "ascending_edge"
cf.sim.load_data()

t = cf.sim.solutions.t[:]
g, x, y, z, ux, uy, uz = cf.sim.solutions.as_tuple()
tg = t[:, np.newaxis] * np.ones_like(x)

_, _, _, B0_x, _, B0_z = cf.model.background_field(
    tg, x, y, z, *cf.model.background_field_args
)
V_para, V_perp, W, A = cartesian_to_FAC(g, ux, uy, uz, B0_x, B0_z, cf.model)
W0 = W[0, :]
particle_colors = mpl.colormaps.get_cmap("cet_rainbow4")(
    np.linspace(0, 1, Np := W0.size)
)


def H_surface(alpha, Vi_para):
    g0 = 1 / np.sqrt(1 - (Vi_para / cf.c.code) ** 2)
    _, _, _, B0_x, _, B0_z = cf.model.background_field(
        0.0, cf.xw.code, 0.0, cf.zw.code, *cf.model.background_field_args
    )
    wce = np.abs(cf.qe.code / cf.me.code) * np.sqrt(B0_x**2 + B0_z**2)
    w, k = cf.model.dispersion_relation(wce, cf.w_wce, cf.wpe0.code, cf.c.code)
    V_phase = w / k

    H0 = g0 * (1 - V_phase * Vi_para / cf.c.code**2)
    R = np.sqrt(
        H0**2
        - 1
        + (V_phase / cf.c.code) ** 2 / (H0**2 + (V_phase / cf.c.code) ** 2)
    )
    vz = (V_phase / cf.c.code) / (
        H0**2 + (V_phase / cf.c.code) ** 2
    ) + R * np.cos(alpha) / np.sqrt(H0**2 + (V_phase / cf.c.code) ** 2)
    vp = R * np.sin(alpha) / H0
    return (
        cf.V_factor.code_to_user(vz).to("1000 km/s"),
        cf.V_factor.code_to_user(vp).to("1000 km/s"),
    )


fig, ax = mu.plt.subplots(1, 1)

for ip in range(Np):
    ax.plot(
        V_para[:, ip].user.value,
        V_perp[:, ip].user.value,
        c=particle_colors[ip],
    )

kw = dict(ec="k", s=20, zorder=9)
ax.scatter(V_para[0, :].user.value, V_perp[0, :].user.value, fc="w", **kw)
ax.scatter(V_para[-1, :].user.value, V_perp[-1, :].user.value, fc="r", **kw)
alpha = np.linspace(0, np.pi, 1000)
for ip in range(V_para.shape[1]):
    vz_h, vp_h = H_surface(alpha, V_para[0, ip].code)
    ax.plot(vz_h, vp_h, "--k", alpha=0.2)

V_perp_arr = np.linspace(0, 0.08, 1000) * cf.c.code
ls = ["--", ":", "-"]
for i, _z in enumerate([-1.0 * cf.R.code, 0.0, 1.0 * cf.R.code]):
    _B0 = cf.B0.code * cf.model.eta(_z, *cf.model.background_field_args)
    _wce = np.abs(cf.qe.code / cf.me.code * _B0)
    _w, _k = cf.model.dispersion_relation(
        _wce, cf.w_wce, cf.wpe0.code, cf.c.code
    )
    _Vr = np.cos(cf.theta) * cf.model.resonant_velocity(
        _wce, cf.w_wce, cf.wpe0.code, cf.c.code
    )
    ax.axvline(
        _Vr * cf.V_factor.user.value, c="k", ls=ls[i], lw=2
    )

ax.set_aspect("equal")

mu.plt.show()

#  z = np.linspace(-3, 3, Nz := 1000) * cf.R.code
#  x = np.linspace(-3, 3, Nx := 1000) * 0.4 * cf.R.code
#  X, Z = np.meshgrid(x, z, indexing="ij")
#  Y = np.zeros_like(X)
#  _, _, _, B0_x, _, B0_z = cf.model.background_field(
#      0.0, X, Y, Z, *cf.model.background_field_args
#  )
#  B0_mag = np.sqrt(B0_x**2 + B0_z**2)
#
#  fig, axes = mu.plt.subplots(2, 1, sharex=True)
#
#  axes[0].streamplot(
#      Z / cf.R.code,
#      X / cf.R.code,
#      B0_z,
#      B0_x,
#      linewidth=0.2 * B0_mag / B0_mag.min(),
#      color="k",
#      density=0.5,
#  )
#  axes[0].set_aspect("equal")
#  axes[0].set_xlim(z[0] / cf.R.code, z[-1] / cf.R.code)
#  axes[0].set_ylim(x[0] / cf.R.code, x[-1] / cf.R.code)
#
#  axes[1].plot(z / cf.R.code, B0_z[Nx // 2, :] / np.abs(cf.B0.code), "-k")
#
#  mu.plt.show()
