import config as cf
import numpy as np
from astropy.units import Quantity
from tvolib import matplotlib_utils as mu

from rls.formula.conversions import cartesian_to_FAC


def H_surface(Vi_para):
    alpha = np.linspace(0, np.pi, 1000)

    g0 = 1 / np.sqrt(1 - (Vi_para / cf.c.code) ** 2)
    _, _, _, B0_x, _, B0_z = cf.model.background_field(
        0.0,
        cf.model.xw.code,
        0.0,
        cf.model.zw.code,
        *cf.model.background_field_args,
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


def get_trap_angle(z_R):
    return np.arcsin(
        np.sqrt(cf.model.eta(z_R * cf.R.code, *cf.model.background_field_args))
    )


# ---- Load data
# -- Array of ICs
cf.sim.name = "array_of_initial_conditions"
cf.sim.load_data()

t = cf.sim.solutions.t[:]
g, x, y, z, ux, uy, uz = cf.sim.solutions.as_tuple()
tg = t[:, np.newaxis] * np.ones_like(x)

_, _, _, B0_x, _, B0_z = cf.model.background_field(
    tg, x, y, z, *cf.model.background_field_args
)
V_para, V_perp, W, A = cartesian_to_FAC(g, ux, uy, uz, B0_x, B0_z, cf.model)
W0 = W[0, :]
dW = W - W0[np.newaxis, :]
particle_colors = mu.mpl.colormaps.get_cmap("cet_rainbow4")(
    np.linspace(0, 1, Np := W0.size)
)

V_perp_arr = np.linspace(0, 0.03, 1000) * cf.c.code
z_arr = np.linspace(-3, 3, 1000) * cf.R.code
B0_z_arr = cf.B0.code * cf.model.eta(z_arr, *cf.model.background_field_args)
wce_arr = np.abs(cf.qe.code / cf.me.code * B0_z_arr)
Vr_arr = cf.model.resonant_velocity(wce_arr, cf.w_wce, cf.wpe0.code, cf.c.code)
trap_angle = np.degrees(np.arcsin(np.sqrt(B0_z_arr / cf.B0.code)))

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

# -- Random ICs
cf.sim.name = "random_initial_conditions"
cf.sim.load_data()

t = cf.sim.solutions.t[:]
g, x, y, z, ux, uy, uz = cf.sim.solutions.as_tuple()
tg = t[:, np.newaxis] * np.ones_like(x)

_, _, _, B0_x, _, B0_z = cf.model.background_field(
    tg, x, y, z, *cf.model.background_field_args
)
V_para, V_perp, W, A = cartesian_to_FAC(g, ux, uy, uz, B0_x, B0_z, cf.model)
Vi_para = V_para[0, :]
Vi_perp = V_perp[0, :]
dW = W[-1, :] - W[0, :]
dA = A[-1, :] - A[0, :]
dW_max = np.abs(dW.user).max().value
dA_max = np.abs(dA.user).max().value

resonant_particles = dA.user < 0
V_perp_bins = np.linspace(0, 10, 200) * cf.V_factor.user.unit
V_para_bins = np.linspace(-12, 0, 200) * cf.V_factor.user.unit
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

# ---- Create figure
fig = mu.plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(15, 3, width_ratios=(4.0, 1.0, 2.5))

# -- Panel a (2D domain)
ax_a = fig.add_subplot(gs[0:3, 0])
cax_a = mu.add_colorbar(ax_a, size="1%")

z = np.linspace(-2, 2, Nz := 1000) * cf.R.code
x = np.linspace(-3, 3, Nx := 1000) * 0.4 * cf.R.code
X, Z = np.meshgrid(x, z, indexing="ij")
Y = np.zeros_like(X)
_, _, _, B0_x, _, B0_z = cf.model.background_field(
    0.0, X, Y, Z, *cf.model.background_field_args
)
B0_mag = np.sqrt(B0_x**2 + B0_z**2)
Ew_x, Ew_y, Ew_z, Bw_x, Bw_y, Bw_z = cf.model.wave_field(
    0.0, X, Y, Z, B0_x, B0_z, *cf.model.wave_field_args
)

ax_a.streamplot(
    Z / cf.R.code,
    X / cf.R.code,
    B0_z,
    B0_x,
    linewidth=0.2 * B0_mag / B0_mag.min(),
    color="k",
    density=0.5,
)
im = ax_a.pcolormesh(
    Z / cf.R.code,
    X / cf.R.code,
    Bw_x / cf.B0.code,
    cmap="seismic",
    vmin=-0.06,
    vmax=0.06,
)
cb_a = fig.colorbar(im, cax=cax_a, ticks=[-0.05, 0, 0.05])
cb_a.set_label("$B_{w,x}/B_0$")

ax_a.set_ylim(x[0] / cf.R.code, x[-1] / cf.R.code)
ax_a.locator_params(axis="y", nbins=5)
ax_a.set_ylabel("$x/R$")
ax_a.arrow(
    -1.8,
    1.45,
    -0.15,
    0,
    color="k",
    linewidth=1,
    head_width=0.1,
    clip_on=False,
)
ax_a.text(-1.75, 1.35, "SW", clip_on=False)
ax_a.arrow(
    1.8,
    1.45,
    0.15,
    0,
    color="k",
    linewidth=1,
    head_width=0.1,
    clip_on=False,
)
ax_a.text(1.3, 1.35, "ASW", clip_on=False)
mu.add_panel_label(ax_a, 0.02, 0.93, "(a)", ha="left", va="top")

# -- Panel b (1D domain)
ax_b_l = fig.add_subplot(gs[3:6, 0])
ax_b_r = ax_b_l.twinx()
ax_b_r.plot(
    Z[Nx // 2, :] / cf.R.code,
    Bw_x[Nx // 2, :] / cf.B0.code,
    "-r",
)
ax_b_l.arrow(-0.4, -0.2, -0.5, 0.0, color="r", linewidth=2, head_width=0.02)
ax_b_l.text(-0.7, -0.17, "$\\vec{k}$", color="r")
ax_b_l.plot(
    Z[Nx // 2, :] / cf.R.code,
    B0_z[Nx // 2, :] / np.abs(cf.B0.code),
    "-k",
)
ax_b_l.set_ylabel("$B_{0,z}/B_0$")
for ax in [ax_b_r, ax_b_l]:
    mu.add_colorbar(ax, size="1%").remove()
    ax.set_xticklabels([])
    ax.locator_params(axis="y", nbins=5)

ax_b_l.set_ylim(-1.1, 0)
ax_b_r.set_ylim(-0.07, 0.07)
ax_b_r.set_ylabel("$B_{w,x}/B_0$")
ax_b_r.tick_params(axis="y", colors="r")
ax_b_r.yaxis.label.set_color("r")
mu.add_panel_label(ax_b_l, 0.02, 0.93, "(b)", ha="left", va="top")

# -- Panel c (z vs Vz)
ax_c_l = fig.add_subplot(gs[6:9, 0])
ax_c_r = ax_c_l.twinx()
mu.add_colorbar(ax_c_l, size="1%").remove()
mu.add_colorbar(ax_c_r, size="1%").remove()

for ip in range(data_array["Np"]):
    ax_c_l.plot(
        data_array["z"][:, ip] / cf.R.code,
        data_array["V_para"][:, ip].user.value,
        c=data_array["particle_colors"][ip],
    )

ax_c_l.plot(
    data_array["z_arr"] / cf.R.code,
    data_array["Vr_arr"] * cf.V_factor.user.value,
    "--k",
    lw=2,
    zorder=9,
)

Vticks = np.arange(-12, 1, 4)
Eticks = (
    (0.5 * cf.me.user * Quantity(Vticks, "1000 km/s") ** 2)
    .to("eV")
    .value.astype(np.int64)
    .astype(str)
)
ax_c_l.set_ylim(-12, 0)
ax_c_l.set_yticks(Vticks)
ax_c_l.locator_params(axis="y", nbins=5)
ax_c_l.set_ylabel(f"$V_\\|$ ({cf.V_factor.user.unit:latex_inline})")
mu.add_panel_label(ax_c_l, 0.02, 0.93, "(c)", ha="left", va="top")

ax_c_r.set_ylim(-12, 0)
ax_c_r.set_yticks(Vticks)
ax_c_r.set_yticklabels(Eticks)
ax_c_r.set_ylabel("$W_\\|$ (eV)")

# -- Panel d (Pitch angle)
ax_d = fig.add_subplot(gs[9:12, 0])
mu.add_colorbar(ax_d, size="1%").remove()

ax_d.fill_between(
    data_array["z_arr"] / cf.R.code,
    data_array["trap_angle"],
    color="k",
    alpha=0.2,
)
for ip in range(data_array["Np"]):
    ax_d.plot(
        data_array["z"][:, ip] / cf.R.code,
        np.degrees(data_array["A"][:, ip].user.value),
        c=data_array["particle_colors"][ip],
    )

ax_d.locator_params(axis="y", nbins=5)
ax_d.set_ylabel("$\\alpha$ (deg)")
ax_d.set_ylim(90, 185)
ax_d.set_yticks(np.arange(90, 181, 30))
ax_d.text(-0.7, 100, "Trap")
ax_d.text(-1.7, 110, "Free")
mu.add_panel_label(ax_d, 0.02, 0.93, "(d)", ha="left", va="top")

# -- Energization
ax_e = fig.add_subplot(gs[12:15, 0])
mu.add_colorbar(ax_e, size="1%").remove()

for ip in range(data_array["Np"]):
    ax_e.plot(
        data_array["z"][:, ip] / cf.R.code,
        data_array["dW"][:, ip].user.value,
        c=data_array["particle_colors"][ip],
    )

ax_e.set_ylim(-10, 35)
ax_e.set_yticks(np.arange(-10, 31, 10))
ax_e.locator_params(axis="y", nbins=5)
ax_e.set_xlabel("$z/R$")
ax_e.set_ylabel(f"$\\Delta W$ ({cf.W_factor.user.unit:latex_inline})")
mu.add_panel_label(ax_e, 0.02, 0.93, "(e)", ha="left", va="top")

# -- Velocity space
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
for ip in range(data_array["V_para"].shape[1]):
    vz_h, vp_h = H_surface(data_array["V_para"][0, ip].code)
    ax_f.plot(vz_h, vp_h, "--k", alpha=0.2)

ax_f.set_ylim(0, 10)
ax_f.set_ylabel(f"$V_{{\\perp}}$ ({cf.V_factor.user.unit:latex_inline})")
ax_f.arrow(-4, 8, -2, 0.0, color="k", linewidth=2, head_width=0.1)
ax_f.text(-6, 8.5, "$\\vec{F}_m$")
mu.add_panel_label(ax_f, 0.05, 0.95, "(f)", ha="left", va="top")

# -- Mean energization
ax_g = fig.add_subplot(gs[5:10, 2])
cax_g = mu.add_colorbar(ax_g)

im = ax_g.pcolormesh(
    data_random["Vg_para"].value,
    data_random["Vg_perp"].value,
    data_random["dW_mean"],
    cmap="cet_rainbow4",
)
cb_g = fig.colorbar(im, cax=cax_g)
cb_g.set_label(f"$\\Delta W$ ({cf.W_factor.user.unit:latex_inline})")
ax_g.set_ylabel(f"$V_{{\\perp,0}}$ ({cf.V_factor.user.unit:latex_inline})")
mu.add_panel_label(ax_g, 0.05, 0.95, "(g)", ha="left", va="top")

# Mean scattering
ax_h = fig.add_subplot(gs[10:15, 2])
cax_h = mu.add_colorbar(ax_h)

im = ax_h.pcolormesh(
    data_random["Vg_para"].value,
    data_random["Vg_perp"].value,
    data_random["dA_mean"],
    cmap="cet_rainbow4_r",
)
cb_h = fig.colorbar(im, cax=cax_h, ticks=np.arange(-5, -60, -10))
cb_h.set_label("$\\Delta\\alpha$ (deg)")
ax_h.set_ylabel(f"$V_{{\\perp,0}}$ ({cf.V_factor.user.unit:latex_inline})")
ax_h.set_xlabel(f"$V_{{\\|,0}}$ ({cf.V_factor.user.unit:latex_inline})")
mu.add_panel_label(ax_h, 0.05, 0.95, "(h)", ha="left", va="top")

# -- Final format
wave_region_ls = ["-", "--"]
wave_region_location = np.array([-1, 1]) * cf.R.code
left_panels = [ax_a, ax_b_l, ax_b_r, ax_c_l, ax_d, ax_e]
right_panels = [ax_f, ax_g, ax_h]
for i, ax in enumerate(left_panels):
    ax.axvline(-1.0, ls=wave_region_ls[0], lw=1)
    ax.axvline(1.0, ls=wave_region_ls[1], lw=1)
    ax.set_xlim(-2, 2)
    ax.locator_params(axis="x", nbins=5)
    if i != len(left_panels) - 1:
        ax.set_xticklabels([])

V_perp_arr = np.linspace(0, 0.08, 1000) * cf.c.code
for i, _z in enumerate(wave_region_location):
    _B0 = cf.B0.code * cf.model.eta(_z, *cf.model.background_field_args)
    _wce = np.abs(cf.qe.code / cf.me.code * _B0)
    _w, _k = cf.model.dispersion_relation(
        _wce, cf.w_wce, cf.wpe0.code, cf.c.code
    )
    _Vr = np.cos(cf.theta) * cf.model.resonant_velocity(
        _wce, cf.w_wce, cf.wpe0.code, cf.c.code
    )
    _dV = cf.model.resonant_width(
        V_perp_arr,
        cf.model.Bw.code,
        cf.model.B0.code,
        _wce,
        cf.w_wce,
        cf.wpe0.code,
        cf.c.code,
    )
    for j, ax in enumerate(right_panels):
        ax.axvline(
            _Vr * cf.V_factor.user.value,
            c="k" if j == 0 else "w",
            ls=wave_region_ls[i],
            lw=1,
        )

V_para_arr = np.linspace(-10, 0, 1000)
for i, ax in enumerate(right_panels):
    ax.set_xlim(-10, 0)
    ax.set_ylim(0, 10)
    ax.set_yticks(np.arange(0, 10, 2))
    ax.locator_params(axis="both", nbins=5)
    ax.fill_between(
        V_para_arr,
        -V_para_arr * np.tan(get_trap_angle(1)),
        10,
        color="k" if ax == ax_f else "w",
        alpha=0.2,
    )
    if i != len(right_panels) - 1:
        ax.set_xticklabels([])


fig.align_ylabels([ax_a, ax_b_l, ax_c_l, ax_d, ax_e])
fig.align_ylabels(right_panels)
fig.subplots_adjust(top=0.97, left=0.09, right=0.92, bottom=0.08)
fig.savefig(
    "../manuscript/assets/setup_and_single_particle_trajectories_hires.png",
    dpi=300,
)
fig.savefig(
    "../manuscript/assets/setup_and_single_particle_trajectories_lores.png",
    dpi=100,
)
mu.plt.show()
