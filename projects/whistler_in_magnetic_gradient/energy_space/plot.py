import astropy.units as u
import config as cf
import numpy as np
import zarr
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.units import Quantity
from tvolib import matplotlib_utils as mu

from rls.formula.physics import f_kappa, f_maxwellian
from rls.io import data_dir, work_dir


def get_data(name):
    nc = Quantity(75.63, "cm-3")
    # nh = Quantity(96.46, "cm-3")
    nh = Quantity(50.46, "cm-3")
    ns = Quantity(5.35, "cm-3")
    Vthc = Quantity(3200, "km/s")
    Vthh = Quantity(2560, "km/s")
    Vths_para = Quantity(4450, "km/s")
    Vths_perp = Quantity(3320, "km/s")
    V_drift = Quantity(-4130, "km/s")
    kh = 4.31
    ks = 9.13

    where = lambda var: data_dir / f"{name}/processed/{var}"
    Vi_para = Quantity(zarr.load(where("Vi_para")), "1000 km/s")
    Vi_perp = Quantity(zarr.load(where("Vi_perp")), "1000 km/s")
    Wi = Quantity(zarr.load(where("Wi")), "eV")
    Ai = Quantity(zarr.load(where("Ai")), "rad").to("deg")
    Wf = Quantity(zarr.load(where("Wf")), "eV")
    Af = Quantity(zarr.load(where("Af")), "rad").to("deg")
    fc = f_maxwellian(
        Vi_para,
        Vi_perp,
        n=nc,
        Vth_para=Vthc,
        Vth_perp=Vthc,
        V_drift=Quantity(0, "km/s"),
    )
    fh = f_kappa(
        Vi_para,
        Vi_perp,
        n=nh,
        Vth_para=Vthh,
        Vth_perp=Vthh,
        kappa=kh,
        V_drift=Quantity(0, "km/s"),
    )
    fs = f_kappa(
        Vi_para,
        Vi_perp,
        n=ns,
        Vth_para=Vths_para,
        Vth_perp=Vths_perp,
        kappa=ks,
        V_drift=V_drift,
    )
    j = (
        ((Vi_para**2 + Vi_perp**2) ** 2 / 2 * (fc + fh + fs))
        .to(u.Unit("cm-2 s-1"))
        .value
    )

    # Base counts
    Hic = np.histogram2d(Ai, Wi, bins=(A_bins, W_bins))[0]
    Hfc = np.histogram2d(Af, Wf, bins=(A_bins, W_bins))[0]

    # Weighted DF
    Hi = np.histogram2d(Ai, Wi, bins=(A_bins, W_bins), weights=j)[0]
    Hf = np.histogram2d(Af, Wf, bins=(A_bins, W_bins), weights=j)[0]

    fi = Hi / Hic
    ff = Hf / Hfc
    return fi, ff


def plot_spectrum(ax, name, Bh_B0, w_wce):
    fi, ff = get_data(name)
    model = cf.model
    units = model.units
    model.Bh = Bh_B0 * units.B_factor
    model.w_wce = w_wce
    q = units.electron.q.code
    m = units.electron.m.code
    c = units.c.code
    wpe0 = units.electron.wp(model.n0, units.eps0).code
    xw = model.xw.code
    zw = model.zw.code
    sw = model.sw.code

    ax.contour(
        Ag.value,
        Wg.value,
        convolve(fi.value, kernel, **kkw),
        linestyles="--",
        **ckw,
    )
    ax.contour(
        Ag.value,
        Wg.value,
        convolve(ff.value, kernel, **kkw),
        linestyles="-",
        **ckw,
    )
    im = ax.pcolormesh(
        Ag.value,
        Wg.value,
        convolve(ff.value, kernel, **kkw),
        cmap="cet_rainbow4",
        norm=mu.mplc.LogNorm(1e6, 1e10),
    )

    def A_res(W, x, z):
        g = 1 + (W / units.W_factor.user).decompose()
        V = np.sqrt(1 - 1 / g**2)
        _, _, _, B0_x, _, B0_z = model.background_field(
            0.0, x, 0.0, z, *model.background_field_args
        )
        wce = np.abs(q / m) * np.sqrt(B0_x**2 + B0_z**2)
        w, k = model.dispersion_relation(wce, w_wce, wpe0, c)
        return np.degrees(np.arccos(wce * (g * w_wce - 1) / (k * V)))

    Am_arr = A_res(W_arr, xw, zw - 2 * sw)
    Ap_arr = A_res(W_arr, xw, zw + 2 * sw)
    Am_arr[np.isnan(Am_arr)] = Quantity(180, "deg")
    Ap_arr[np.isnan(Ap_arr)] = Quantity(180, "deg")
    ax.plot(Ap_arr.value, W_arr.value, color="w", ls="--", lw=2)
    ax.plot(Am_arr.value, W_arr.value, color="w", ls="-", lw=2)
    #  ax.fill_betweenx(
    #      W_arr.value, Ap_arr.value, Am_arr.value, color="w", alpha=0.2,
    #  )
    return im


def plot_profile(ax_l, ax_r, name, Bh_B0, w_wce):
    model = cf.model
    units = model.units
    model.Bh = Bh_B0 * units.B_factor
    model.w_wce = w_wce
    R = model.R.code
    B0 = np.abs(model.B0.code)

    z = np.linspace(-3, 3, 1000) * model.R.code
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    _, _, _, B0_x, _, B0_z = model.background_field(
        0.0, x, y, z, *model.background_field_args
    )
    B0_mag = np.sqrt(B0_x**2 + B0_z**2)
    Ew_x, Ew_y, Ew_z, Bw_x, Bw_y, Bw_z = model.wave_field(
        0.0,
        x,
        y,
        z,
        B0_x,
        B0_z,
        *model.wave_field_args,
    )

    ax_r.plot(z / R, Bw_x / B0, "-r")
    ax_l.plot(z / R, B0_mag / B0, "-k")
    mu.draw_arrows(ax_l, z[::-1] / R, B0_mag[::-1] / B0, number_of_arrows=4)

    ax_l.set_xlim(z[0] / R, z[-1] / R)
    ax_l.set_ylim(0, 1.2)
    ax_l.set_yticks(np.arange(0, 1.1, 0.5))
    ax_l.set_xlabel("$z/R$")
    ax_l.set_ylabel("$|B|/B_0$")

    ax_r.set_ylim(-0.012, 0.012)
    ax_r.set_yticks([-0.01, 0, 0.01])
    ax_r.set_ylabel("$B_{w,x}/B_0$")
    ax_r.tick_params(axis="y", colors="r")
    ax_r.yaxis.label.set_color("r")


W_bins = Quantity(np.logspace(np.log10(50), np.log10(2e3), 100), "eV")
A_bins = Quantity(np.arange(0, 181, 1), "deg")
Ag, Wg = np.meshgrid(A_bins[:-1], W_bins[:-1], indexing="ij")
W_arr = Quantity(np.logspace(np.log10(50), np.log10(2e3), 1000), "eV")

ckw = dict(
    norm=mu.mplc.LogNorm(),
    levels=np.logspace(6, 10, 10),
    zorder=9,
    linewidths=1,
    colors="k",
)
kernel = Gaussian2DKernel(4)
kkw = dict(boundary="extend")

fig = mu.plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(
    6,
    8,
    height_ratios=(1, 0.25, 1, 1, 1, 1),
    hspace=0.4,
    wspace=0.05,
    left=0.08,
    right=0.92,
    bottom=0.07,
    top=0.96,
)

ax_a_l = fig.add_subplot(gs[0, :])
ax_a_r = ax_a_l.twinx()
mu.add_colorbar(ax_a_l).remove()
mu.add_colorbar(ax_a_r).remove()
plot_profile(ax_a_l, ax_a_r, "fig_energy_space_w_005_Bw_001_Bh_08", -0.8, 0.05)
mu.add_panel_label(ax_a_l, 0.01, 0.9, "(a)", ha="left", va="top")
ax_a_l.arrow(-2, 0.4, -0.5, 0.0, color="r", linewidth=2, head_width=0.02)
ax_a_l.text(-2.3, 0.10, "$\\vec{k}$", color="r")
ax_a_l.arrow(
    -2.5,
    1.42,
    -0.4,
    0,
    color="k",
    linewidth=1,
    head_width=0.1,
    clip_on=False,
)
ax_a_l.text(-2.4, 1.35, "SW", clip_on=False)
ax_a_l.arrow(
    2.5,
    1.42,
    0.4,
    0,
    color="k",
    linewidth=1,
    head_width=0.1,
    clip_on=False,
)
ax_a_l.text(2.4, 1.35, "ASW", clip_on=False, ha="right")

cb_kw = dict(size="5%", pad=0.08)

ax_b1 = fig.add_subplot(gs[2:4, 0:2])
mu.add_colorbar(ax_b1, **cb_kw).remove()
im = plot_spectrum(ax_b1, "fig_energy_space_w_005_Bw_001_Bh_03", -0.3, 0.05)
mu.add_panel_label(ax_b1, 0.05, 0.95, "(b-1)", ha="left", va="top")

ax_c1 = fig.add_subplot(gs[2:4, 2:4])
mu.add_colorbar(ax_c1, **cb_kw).remove()
im = plot_spectrum(ax_c1, "fig_energy_space_w_010_Bw_001_Bh_03", -0.3, 0.1)
mu.add_panel_label(ax_c1, 0.05, 0.95, "(b-2)", ha="left", va="top")

ax_d1 = fig.add_subplot(gs[2:4, 4:6])
mu.add_colorbar(ax_d1, **cb_kw).remove()
im = plot_spectrum(ax_d1, "fig_energy_space_w_015_Bw_001_Bh_03", -0.3, 0.15)
mu.add_panel_label(ax_d1, 0.05, 0.95, "(b-3)", ha="left", va="top")

ax_e1 = fig.add_subplot(gs[2:4, 6:8])
mu.add_colorbar(ax_e1, **cb_kw).remove()
im = plot_spectrum(ax_e1, "fig_energy_space_w_020_Bw_001_Bh_03", -0.3, 0.2)
mu.add_panel_label(ax_e1, 0.05, 0.95, "(b-4)", ha="left", va="top")

ax_b2 = fig.add_subplot(gs[4:6, 0:2])
mu.add_colorbar(ax_b2, **cb_kw).remove()
im = plot_spectrum(ax_b2, "fig_energy_space_w_005_Bw_001_Bh_08", -0.8, 0.05)
mu.add_panel_label(ax_b2, 0.05, 0.95, "(c-1)", ha="left", va="top")

ax_c2 = fig.add_subplot(gs[4:6, 2:4])
mu.add_colorbar(ax_c2, **cb_kw).remove()
im = plot_spectrum(ax_c2, "fig_energy_space_w_010_Bw_001_Bh_08", -0.8, 0.1)
mu.add_panel_label(ax_c2, 0.05, 0.95, "(c-2)", ha="left", va="top")

ax_d2 = fig.add_subplot(gs[4:6, 4:6])
mu.add_colorbar(ax_d2, **cb_kw).remove()
im = plot_spectrum(ax_d2, "fig_energy_space_w_015_Bw_001_Bh_08", -0.8, 0.15)
mu.add_panel_label(ax_d2, 0.05, 0.95, "(c-3)", ha="left", va="top")

ax_e2 = fig.add_subplot(gs[4:6, 6:8])
cax = mu.add_colorbar(ax_e2, **cb_kw)
im = plot_spectrum(ax_e2, "fig_energy_space_w_020_Bw_001_Bh_08", -0.8, 0.2)
mu.add_panel_label(ax_e2, 0.05, 0.95, "(c-4)", ha="left", va="top")
cb = fig.colorbar(im, cax=cax)
cb.set_label("eV/(cm$^2$ s sr eV)")


def get_trap_angle(z_R, Bh_B0):
    cf.model.Bh = Bh_B0 * cf.model.B0
    B0_z = cf.model.B0.code * cf.model.eta(
        z_R * cf.R.code, *cf.model.background_field_args
    )
    return np.degrees(np.arcsin(np.sqrt(B0_z / cf.model.B0.code)))


kw = dict(pad=10, fontsize="medium")
ax_b1.set_title(r"$\omega/\Omega_{e}=0.05,B_h/B_0=0.3$", **kw)
ax_c1.set_title(r"$\omega/\Omega_{e}=0.1,B_h/B_0=0.3$", **kw)
ax_d1.set_title(r"$\omega/\Omega_{e}=0.15,B_h/B_0=0.3$", **kw)
ax_e1.set_title(r"$\omega/\Omega_{e}=0.2,B_h/B_0=0.3$", **kw)
ax_b2.set_title(r"$\omega/\Omega_{e}=0.05,B_h/B_0=0.8$", **kw)
ax_c2.set_title(r"$\omega/\Omega_{e}=0.1,B_h/B_0=0.8$", **kw)
ax_d2.set_title(r"$\omega/\Omega_{e}=0.15,B_h/B_0=0.8$", **kw)
ax_e2.set_title(r"$\omega/\Omega_{e}=0.2,B_h/B_0=0.8$", **kw)
axes = np.array([[ax_b1, ax_c1, ax_d1, ax_e1], [ax_b2, ax_c2, ax_d2, ax_e2]])
skw = dict(marker="X", s=200, zorder=999, clip_on=False, ec="w", fc="k")
for i, j in np.ndindex(axes.shape):
    ax = axes[i, j]
    ax.locator_params(axis="x", nbins=5)
    ax.set_facecolor("silver")
    ax.set_yscale("log")
    ax.set_ylim(50, 2e3)
    ax.set_xlim(0, 180)
    if j == 0:
        ax.set_ylabel("$W$ (eV)")
    else:
        ax.set_yticklabels([])
    if i == 1:
        ax.set_xlabel("$\\alpha$ (deg)")
        ax.scatter([180 - get_trap_angle(1, 0.8)], [50], **skw)
    else:
        ax.set_xticklabels([])
        ax.scatter([180 - get_trap_angle(1, 0.3)], [50], **skw)

fig.align_ylabels([ax_a_l, ax_b1, ax_c1])
fig.savefig(
    "../manuscript/assets/weighted_energy_space_hires.png",
    dpi=300,
)
fig.savefig(
    "../manuscript/assets/weighted_energy_space_lores.png",
    dpi=100,
)
mu.plt.show()
