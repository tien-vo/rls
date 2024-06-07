import astropy.units as u
import colorcet
import config as cf
import numpy as np
import zarr
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.units import Quantity
from rls.formula.physics import f_kappa, f_maxwellian
from rls.io import data_dir, work_dir
from tvolib import mpl_utils as mu


def get_data(name):
    nc = Quantity(75.63, "cm-3")
    nh = Quantity(96.46, "cm-3")
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

    Am_arr = A_res(W_arr, xw, zw - sw)
    Ap_arr = A_res(W_arr, xw, zw + sw)
    Am_arr[np.isnan(Am_arr)] = Quantity(180, "deg")
    Ap_arr[np.isnan(Ap_arr)] = Quantity(180, "deg")
    ax.fill_betweenx(
        W_arr.value, Ap_arr.value, Am_arr.value, color="w", alpha=0.2,
    )
    return im


def plot_profile(ax_l, ax_r, name, Bh_B0, w_wce):
    model = cf.model
    units = model.units
    model.Bh = Bh_B0 * units.B_factor
    model.w_wce = w_wce
    R = model.R.code
    B0 = np.abs(model.B0.code)

    z = np.linspace(-3, 3, Nz := 1000) * model.R.code
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
    mu.draw_arrows(ax_l, z[::-1] / R, B0_mag[::-1] / B0, N=4)

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

fig = mu.plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(
    6, 6,
    height_ratios=(1, 0.25, 1, 1, 1, 1),
    hspace=0.4, wspace=0.05,
    left=0.08, right=0.92, bottom=0.07, top=0.96,
)

ax_a_l = fig.add_subplot(gs[0, :])
ax_a_r = ax_a_l.twinx()
mu.add_colorbar(ax_a_l).remove()
mu.add_colorbar(ax_a_r).remove()
plot_profile(ax_a_l, ax_a_r, "fig_energy_space_w_005_Bw_001_Bh_08", -0.8, 0.05)
mu.add_text(ax_a_l, 0.01, 0.9, "(a)", ha="left", va="top")
ax_a_l.arrow(-2, 0.4, -0.5, 0.0, color="r", linewidth=2, head_width=0.02)
ax_a_l.text(-2.3, 0.10, "$\\vec{k}$", color="r")
ax_a_l.arrow(
    -2.5, 1.42, -0.4, 0,
    color="k",
    linewidth=1,
    head_width=0.1,
    clip_on=False,
)
ax_a_l.text(-2.4, 1.35, "SW", clip_on=False)
ax_a_l.arrow(
    2.5, 1.42, 0.4, 0,
    color="k",
    linewidth=1,
    head_width=0.1,
    clip_on=False,
)
ax_a_l.text(2.4, 1.35, "ASW", clip_on=False, ha="right")

ax_b = fig.add_subplot(gs[2:4, 0:2])
mu.add_colorbar(ax_b).remove()
im = plot_spectrum(ax_b, "fig_energy_space_w_005_Bw_001_Bh_03", -0.3, 0.05)
mu.add_text(ax_b, 0.05, 0.95, "(b-1)", ha="left", va="top")

ax_c = fig.add_subplot(gs[2:4, 2:4])
mu.add_colorbar(ax_c).remove()
im = plot_spectrum(ax_c, "fig_energy_space_w_010_Bw_001_Bh_03", -0.3, 0.1)
mu.add_text(ax_c, 0.05, 0.95, "(b-2)", ha="left", va="top")

ax_d = fig.add_subplot(gs[2:4, 4:6])
mu.add_colorbar(ax_d).remove()
im = plot_spectrum(ax_d, "fig_energy_space_w_015_Bw_001_Bh_03", -0.3, 0.15)
mu.add_text(ax_d, 0.05, 0.95, "(b-3)", ha="left", va="top")

ax_e = fig.add_subplot(gs[4:6, 0:2])
mu.add_colorbar(ax_e).remove()
im = plot_spectrum(ax_e, "fig_energy_space_w_005_Bw_001_Bh_08", -0.8, 0.05)
mu.add_text(ax_e, 0.05, 0.95, "(c-1)", ha="left", va="top")

ax_f = fig.add_subplot(gs[4:6, 2:4])
mu.add_colorbar(ax_f).remove()
im = plot_spectrum(ax_f, "fig_energy_space_w_010_Bw_001_Bh_08", -0.8, 0.1)
mu.add_text(ax_f, 0.05, 0.95, "(c-2)", ha="left", va="top")

ax_g = fig.add_subplot(gs[4:6, 4:6])
cax = mu.add_colorbar(ax_g)
im = plot_spectrum(ax_g, "fig_energy_space_w_015_Bw_001_Bh_08", -0.8, 0.15)
mu.add_text(ax_g, 0.05, 0.95, "(c-3)", ha="left", va="top")
cb = fig.colorbar(im, cax=cax)
cb.set_label("eV/(cm$^2$ s sr eV)")

ax_b.set_title(r"$\omega/\Omega_{e}=0.05,B_h/B_0=0.3$", pad=10)
ax_c.set_title(r"$\omega/\Omega_{e}=0.1,B_h/B_0=0.3$", pad=10)
ax_d.set_title(r"$\omega/\Omega_{e}=0.15,B_h/B_0=0.3$", pad=10)
ax_e.set_title(r"$\omega/\Omega_{e}=0.05,B_h/B_0=0.8$", pad=10)
ax_f.set_title(r"$\omega/\Omega_{e}=0.1,B_h/B_0=0.8$", pad=10)
ax_g.set_title(r"$\omega/\Omega_{e}=0.15,B_h/B_0=0.8$", pad=10)
axes = np.array([
    [ax_b, ax_c, ax_d],
    [ax_e, ax_f, ax_g]
])
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
    else:
        ax.set_xticklabels([])

fig.align_ylabels([ax_a_l, ax_b, ax_f])
fig.savefig(work_dir / "plots" / "weighted_energy_space.png", dpi=600)
fig.savefig(work_dir / "plots" / "weighted_energy_space_lowres.png")
