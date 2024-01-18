import colorcet
import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rls.formula.conversions import cartesian_to_FAC
from rls.io import work_dir
from rls.models import WhistlerAtHyperbolicGradientModel
from rls.simulation import Simulation
from tvolib import mpl_utils as mu

sim = Simulation(
    model=WhistlerAtHyperbolicGradientModel(
        Bw=0.05,
        w_wce=0.1,
        sw=1.75,
        Bh=-0.8,
        B0=-1.0,
        theta=0.0,
    )
)
model = sim.model
units = model.units

c = units.c
R = model.R
B0 = model.B0
Bw = model.Bw
w_wce = model.w_wce
theta = model.theta
xw = model.xw
zw = model.zw
sw = model.sw
qe = units.electron.q
me = units.electron.m
wpe0 = units.electron.wp(model.n0, units.eps0)
wce0 = units.electron.wc(model.B0)
V_factor = (units.L_factor / units.T_factor).to("1000 km/s")
W_factor = units.W_factor

sim.name = "fig_setup_array_IC"
sim.load_data()

t = sim.solutions.t[:]
g, x, y, z, ux, uy, uz = sim.solutions.as_tuple()
tg = t[:, np.newaxis] * np.ones_like(x)

args = model.background_field_args
_, _, _, B0_x, _, B0_z = model.background_field(tg, x, y, z, *args)
V_para, V_perp, W, A = cartesian_to_FAC(g, ux, uy, uz, B0_x, B0_z, model)
W0 = W[0, :]
dW = W - W0[np.newaxis, :]
particle_colors = mpl.colormaps.get_cmap("cet_rainbow4")(
    np.linspace(0, 1, Np := W0.size)
)

V_perp_arr = np.linspace(0, 0.03, 1000) * c.code
z_arr = np.linspace(-3, 3, 1000) * R.code
B0_z_arr = B0.code * model.eta(z_arr, *model.background_field_args)
wce_arr = np.abs(qe.code / me.code * B0_z_arr)
Vr_arr = model.resonant_velocity(wce_arr, w_wce, wpe0.code, c.code)
trap_angle = np.degrees(np.arcsin(np.sqrt(B0_z_arr / B0.code)))

data_array = dict(
    ux=ux,
    uy=uy,
    z=z,
    V_para=V_para,
    V_perp=V_perp,
    dW=dW,
    A=A,
    z_arr=z_arr,
    Vr_arr=Vr_arr,
    B0_z_arr=B0_z_arr,
    trap_angle=180 - trap_angle,
    Np=Np,
    particle_colors=particle_colors,
)

sim.name = "fig_setup_random_IC"
sim.load_data()

t = sim.solutions.t[:]
g, x, y, z, ux, uy, uz = sim.solutions.as_tuple()
tg = t[:, np.newaxis] * np.ones_like(x)

args = model.background_field_args
_, _, _, B0_x, _, B0_z = model.background_field(tg, x, y, z, *args)
wce = np.abs(qe.code / me.code * np.sqrt(B0_x**2 + B0_z**2))
w, k = model.dispersion_relation(wce, w_wce, wpe0.code, c.code)
V_para, V_perp, W, A = cartesian_to_FAC(g, ux, uy, uz, B0_x, B0_z, model)
Vi_para = V_para[0, :]
Vi_perp = V_perp[0, :]
dW = W[-1, :] - W[0, :]
dA = A[-1, :] - A[0, :]
dW_max = np.abs(dW.user).max().value
dA_max = np.abs(dA.user).max().value

resonant_particles = dA.user < 0
V_perp_bins = np.linspace(0, 10, 200) * V_factor.user.unit
V_para_bins = np.linspace(-12, 0, 200) * V_factor.user.unit
Vg_para, Vg_perp = np.meshgrid(
    V_para_bins[:-1], V_perp_bins[:-1], indexing="ij"
)
H_2d = np.histogram2d(
    Vi_para.user,
    Vi_perp.user,
    bins=(V_para_bins, V_perp_bins),
)[0]
H_dW = np.histogram2d(
    Vi_para[resonant_particles].user,
    Vi_perp[resonant_particles].user,
    bins=(V_para_bins, V_perp_bins),
    weights=dW[resonant_particles].user,
)[0]
H_dA = np.histogram2d(
    Vi_para[resonant_particles].user,
    Vi_perp[resonant_particles].user,
    bins=(V_para_bins, V_perp_bins),
    weights=np.degrees(dA[resonant_particles].user),
)[0]
dW_mean = (H_dW / H_2d).value
dA_mean = (H_dA / H_2d).value
dW_mean[np.isnan(dW_mean)] = 0.0
dA_mean[np.isnan(dA_mean)] = 0.0

data_random = dict(
    Vg_para=Vg_para,
    Vg_perp=Vg_perp,
    V_para=V_para_bins[:-1],
    V_perp=V_perp_bins[:-1],
    dA_mean=dA_mean,
    dW_mean=dW_mean,
)

fig = mu.plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(15, 3, width_ratios=(4.0, 1.0, 2.5))

# 2D domain
z = np.linspace(-3, 3, Nz := 1000) * R.code
x = np.linspace(-3, 3, Nx := 1000) * 0.4 * R.code
X, Z = np.meshgrid(x, z, indexing="ij")
Y = np.zeros_like(X)
_, _, _, B0_x, _, B0_z = model.background_field(
    0.0, X, Y, Z, *model.background_field_args
)
B0_mag = np.sqrt(B0_x**2 + B0_z**2)
Ew_x, Ew_y, Ew_z, Bw_x, Bw_y, Bw_z = model.wave_field(
    0.0,
    X,
    Y,
    Z,
    B0_x,
    B0_z,
    *model.wave_field_args,
)
ax_a = fig.add_subplot(gs[0:3, 0])
ax_a.streamplot(
    Z / R.code,
    X / R.code,
    B0_z,
    B0_x,
    linewidth=0.2 * B0_mag / B0_mag.min(),
    color="k",
    density=0.5,
)
im = ax_a.pcolormesh(
    Z / R.code,
    X / R.code,
    Bw_x / B0.code,
    cmap="seismic",
    vmin=-0.06,
    vmax=0.06,
)
cax = mu.add_colorbar(ax_a, size="1%")
cb = fig.colorbar(im, cax=cax, ticks=[-0.05, 0, 0.05])
cb.set_label("$B_w/B_0$")
ax_a.set_xlim(z[0] / R.code, z[-1] / R.code)
ax_a.set_ylim(x[0] / R.code, x[-1] / R.code)
ax_a.set_xticklabels([])
ax_a.locator_params(axis="y", nbins=5)
ax_a.set_ylabel("$x/R$")
mu.add_text(ax_a, 0.02, 0.93, "(a)", ha="left", va="top")

# 1D domain
ax_b_l = fig.add_subplot(gs[3:6, 0])
ax_b_r = ax_b_l.twinx()
ax_b_r.plot(
    Z[Nx // 2, :] / R.code, Bw_x[Nx // 2, :] / B0.code, "-b", label="$B_{w,x}$"
)
ax_b_r.plot(
    Z[Nx // 2, :] / R.code, Bw_y[Nx // 2, :] / B0.code, "-r", label="$B_{w,y}$"
)
ax_b_l.arrow(-0.4, -0.2, -0.5, 0.0, color="k", linewidth=2, head_width=0.02)
ax_b_l.text(-0.7, -0.15, "$\\vec{k}$")
ax_b_l.plot(Z[Nx // 2, :] / R.code, B0_z[Nx // 2, :] / np.abs(B0.code), "-k")
ax_b_l.set_ylabel("$B_{0z}/B_0$")
for ax in [ax_b_r, ax_b_l]:
    mu.add_colorbar(ax, size="1%").remove()
    ax.set_xlim(z[0] / R.code, z[-1] / R.code)
    ax.set_xticklabels([])
    ax.locator_params(axis="y", nbins=5)

mu.add_text(ax_b_l, 0.02, 0.93, "(b)", ha="left", va="top")
ax_b_l.set_ylim(-1.0, 0)
ax_b_r.set_ylim(-0.07, 0.07)
ax_b_r.legend(
    loc="lower right",
    frameon=False,
    borderpad=0,
    borderaxespad=0.2,
    handletextpad=0.4,
    handlelength=1.5,
    labelspacing=0.2,
)
ax_b_r.set_ylabel("$B_w/B_0$")

# z vs Vz
ax_c = fig.add_subplot(gs[6:9, 0])
for ip in range(data_array["Np"]):
    ax_c.plot(
        data_array["z"][:, ip] / R.code,
        data_array["V_para"][:, ip].user.value,
        c=data_array["particle_colors"][ip],
    )

mu.add_colorbar(ax_c, size="1%").remove()
ax_c.plot(
    data_array["z_arr"] / R.code,
    data_array["Vr_arr"] * V_factor.user.value,
    "--k",
    lw=2,
    zorder=9,
)
ax_c.set_xlim(z[0] / R.code, z[-1] / R.code)
ax_c.set_ylim(-12, 2)
ax_c.set_xticklabels([])
ax_c.locator_params(axis="y", nbins=5)
ax_c.set_ylabel(f"$V_\\|$ ({V_factor.user.unit:latex_inline})")
mu.add_text(ax_c, 0.02, 0.93, "(c)", ha="left", va="top")

# Pitch angle
ax_d = fig.add_subplot(gs[9:12, 0])
ax_d.fill_between(
    data_array["z_arr"] / R.code,
    data_array["trap_angle"],
    color="k",
    alpha=0.2,
)
for ip in range(data_array["Np"]):
    ax_d.plot(
        data_array["z"][:, ip] / R.code,
        np.degrees(data_array["A"][:, ip].user.value),
        c=data_array["particle_colors"][ip],
    )

mu.add_colorbar(ax_d, size="1%").remove()
ax_d.set_xlim(z[0] / R.code, z[-1] / R.code)
ax_d.set_xticklabels([])
ax_d.locator_params(axis="y", nbins=5)
ax_d.set_ylabel("$\\alpha$ (deg)")
ax_d.set_ylim(90, 180)
ax_d.text(-0.7, 100, "Trap")
ax_d.text(-1.7, 110, "Free")
mu.add_text(ax_d, 0.02, 0.93, "(d)", ha="left", va="top")

# Energization
ax_e = fig.add_subplot(gs[12:15, 0])
for ip in range(data_array["Np"]):
    ax_e.plot(
        data_array["z"][:, ip] / R.code,
        data_array["dW"][:, ip].user.value,
        c=data_array["particle_colors"][ip],
    )

mu.add_colorbar(ax_e, size="1%").remove()
ax_e.set_xlim(z[0] / R.code, z[-1] / R.code)
ax_e.locator_params(axis="y", nbins=5)
ax_e.set_xlabel("$z/R$")
ax_e.set_ylabel(f"$\\Delta W$ ({W_factor.user.unit:latex_inline})")
mu.add_text(ax_e, 0.02, 0.93, "(e)", ha="left", va="top")


# Velocity space
def H_surface(alpha, Vi_para):
    g0 = 1 / np.sqrt(1 - (Vi_para / c.code) ** 2)
    _, _, _, B0_x, _, B0_z = model.background_field(
        0.0, xw.code, 0.0, zw.code, *model.background_field_args
    )
    wce = np.abs(qe.code / me.code) * np.sqrt(B0_x**2 + B0_z**2)
    w, k = model.dispersion_relation(wce, w_wce, wpe0.code, c.code)
    V_phase = w / k

    H0 = g0 * (1 - V_phase * Vi_para / c.code**2)
    R = np.sqrt(
        H0**2 - 1 + (V_phase / c.code) ** 2 / (H0**2 + (V_phase / c.code) ** 2)
    )
    vz = (V_phase / c.code) / (H0**2 + (V_phase / c.code) ** 2) + R * np.cos(
        alpha
    ) / np.sqrt(H0**2 + (V_phase / c.code)**2)
    vp = R * np.sin(alpha) / H0
    return (
        V_factor.code_to_user(vz).to("1000 km/s"),
        V_factor.code_to_user(vp).to("1000 km/s"),
    )


ax_f = fig.add_subplot(gs[0:5, 2])
mu.add_colorbar(ax_f).remove()
for ip in range(data_array["Np"]):
    ax_f.plot(
        data_array["V_para"][:, ip].user.value,
        data_array["V_perp"][:, ip].user.value,
        c=data_array["particle_colors"][ip],
    )

ax_f.scatter(
    data_array["V_para"][0, :].user.value,
    data_array["V_perp"][0, :].user.value,
    fc="w",
    ec="k",
    s=20,
    zorder=9,
)
ax_f.scatter(
    data_array["V_para"][-1, :].user.value,
    data_array["V_perp"][-1, :].user.value,
    fc="r",
    ec="k",
    s=20,
    zorder=9,
)
alpha = np.linspace(0, np.pi, 1000)
for ip in range(data_array["V_para"].shape[1]):
    vz_h, vp_h = H_surface(alpha, data_array["V_para"][0, ip].code)
    ax_f.plot(vz_h, vp_h, "--k", alpha=0.2)

ax_f.set_ylim(0, 10)
ax_f.set_ylabel(f"$V_{{\\perp}}$ ({V_factor.user.unit:latex_inline})")
ax_f.set_xticklabels([])
mu.add_text(ax_f, 0.05, 0.95, "(f)", ha="left", va="top")
ax_f.arrow(-4, 8, -2, 0.0, color="k", linewidth=2, head_width=0.1)
ax_f.text(-6, 8.5, "$\\vec{F}_m$")

""" Not a good idea
ax_inset = inset_axes(ax_f, width="32%", height="32%", loc="upper left")
ax_inset.plot(
    data_array["ux"][:20_000, 12] * V_factor.user.value,
    data_array["uy"][:20_000, 12] * V_factor.user.value,
    "-k",
    lw=0.05,
)
ax_inset.scatter(
    data_array["ux"][[0, -1], 12] * V_factor.user.value,
    data_array["uy"][[0, -1], 12] * V_factor.user.value,
    fc=["w", "r"],
    ec="k",
    s=20,
    zorder=9,
)
ax_inset.yaxis.tick_right()
ax_inset.yaxis.set_label_position("right")
ax_inset.set_xlim(-8, 8)
ax_inset.set_ylim(-8, 8)
ax_inset.locator_params(axis="both", nbins=3)
ax_inset.tick_params(axis="both", labelsize="small")
"""

# Mean energization
ax_g = fig.add_subplot(gs[5:10, 2])
cax = mu.add_colorbar(ax_g)
im = ax_g.pcolormesh(
    data_random["Vg_para"].value,
    data_random["Vg_perp"].value,
    data_random["dW_mean"],
    cmap="cet_rainbow4",
)
cb = fig.colorbar(im, cax=cax)
cb.set_label(f"$\\Delta W$ ({W_factor.user.unit:latex_inline})")
ax_g.set_ylabel(f"$V_{{\\perp,0}}$ ({V_factor.user.unit:latex_inline})")
ax_g.set_xticklabels([])
mu.add_text(ax_g, 0.05, 0.95, "(g)", ha="left", va="top")

# Mean scattering
ax_h = fig.add_subplot(gs[10:15, 2])
cax = mu.add_colorbar(ax_h)
im = ax_h.pcolormesh(
    data_random["Vg_para"].value,
    data_random["Vg_perp"].value,
    data_random["dA_mean"],
    cmap="cet_rainbow4_r",
)
cb = fig.colorbar(im, cax=cax, ticks=np.arange(-5, -60, -10))
cb.set_label("$\\Delta\\alpha$ (deg)")
ax_h.set_ylabel(f"$V_{{\\perp,0}}$ ({V_factor.user.unit:latex_inline})")
ax_h.set_xlabel(f"$V_{{\\|,0}}$ ({V_factor.user.unit:latex_inline})")
mu.add_text(ax_h, 0.05, 0.95, "(h)", ha="left", va="top")

V_perp_arr = np.linspace(0, 0.08, 1000) * c.code
ls = ["-", ":", "--"]
for i, _z in enumerate([-1.0 * R.code, 0.0, 1.0 * R.code]):
    _B0 = B0.code * model.eta(_z, *model.background_field_args)
    _wce = np.abs(qe.code / me.code * _B0)
    _w, _k = model.dispersion_relation(_wce, w_wce, wpe0.code, c.code)
    _Vr = np.cos(theta) * model.resonant_velocity(
        _wce, w_wce, wpe0.code, c.code
    )
    _dV = model.resonant_width(
        V_perp_arr, Bw.code, B0.code, _wce, w_wce, wpe0.code, c.code
    )
    for j, ax in enumerate([ax_f, ax_g, ax_h]):
        ax.axvline(
            _Vr * V_factor.user.value, c="k" if j == 0 else "w", ls=ls[i], lw=2
        )

for ax in [ax_a, ax_b_l, ax_b_r, ax_c, ax_d, ax_e]:
    ax.axvline(-1.0, ls=ls[0])
    ax.axvline(0.0, ls=ls[1])
    ax.axvline(1.0, ls=ls[2])
    ax.set_xlim(-2, 2)
    ax.locator_params(axis="x", nbins=5)

for ax in [ax_f, ax_g, ax_h]:
    ax.set_xlim(-10, 0)
    ax.set_ylim(0, 10)
    ax.set_yticks(np.arange(0, 10, 2))
    ax.locator_params(axis="both", nbins=5)

fig.align_ylabels([ax_a, ax_b_l, ax_c, ax_d, ax_e])
fig.align_ylabels([ax_f, ax_g, ax_h])
fig.align_ylabels([cb.ax, ax_b_r])
fig.subplots_adjust(top=0.98, left=0.09, right=0.92, bottom=0.08)
fig.savefig(work_dir / "plots" / "fig_setup.png", dpi=600)
# mu.plt.show()
