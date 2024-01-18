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

    #  nc = Quantity(240.7, "cm-3")
    #  nh = Quantity(312.1, "cm-3")
    #  ns = Quantity(14.3, "cm-3")
    #  Vthc = Quantity(3212, "km/s")
    #  Vthh = Quantity(2468, "km/s")
    #  Vths_para = Quantity(5344, "km/s")
    #  Vths_perp = Quantity(3383, "km/s")
    #  V_drift = Quantity(-3587, "km/s")
    #  kh = 4.0

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


def plot(ax, name, Bh_B0, w_wce):
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
    ax.plot(Ap_arr, W_arr, "--w", lw=2, zorder=9)
    ax.plot(Am_arr, W_arr, "-w", lw=2, zorder=9)
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
    return im


W_bins = Quantity(np.logspace(np.log10(50), np.log10(2e3), 100), "eV")
A_bins = Quantity(np.arange(0, 181, 1), "deg")
Ag, Wg = np.meshgrid(A_bins[:-1], W_bins[:-1], indexing="ij")
W_arr = Quantity(np.logspace(np.log10(50), np.log10(2e3), 1000), "eV")

fig, axes = mu.plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
ckw = dict(
    norm=mu.mplc.LogNorm(),
    levels=np.logspace(6, 10, 10),
    zorder=9,
    linewidths=1,
    colors="k",
)
kernel = Gaussian2DKernel(4)
kkw = dict(boundary="extend")

mu.add_colorbar(axes[0, 2]).remove()
im = plot(axes[0, 0], "fig_energy_space_Bh_05_Bw_001_A", -0.5, 0.05)
im = plot(axes[0, 1], "fig_energy_space_Bh_05_Bw_001_B", -0.5, 0.10)
im = plot(axes[0, 2], "fig_energy_space_Bh_05_Bw_001_C", -0.5, 0.15)
mu.add_text(axes[0, 0], 0.05, 0.95, "(a-1)", ha="left", va="top")
mu.add_text(axes[0, 1], 0.05, 0.95, "(a-2)", ha="left", va="top")
mu.add_text(axes[0, 2], 0.05, 0.95, "(a-3)", ha="left", va="top")

cax = mu.add_colorbar(axes[1, 2])
im = plot(axes[1, 0], "fig_energy_space_Bh_08_Bw_001_A", -0.8, 0.05)
im = plot(axes[1, 1], "fig_energy_space_Bh_08_Bw_001_B", -0.8, 0.10)
im = plot(axes[1, 2], "fig_energy_space_Bh_08_Bw_001_C", -0.8, 0.15)
cb = fig.colorbar(im, cax=cax)
cb.set_label("eV/(cm$^2$ s sr eV)")
mu.add_text(axes[1, 0], 0.05, 0.95, "(b-1)", ha="left", va="top")
mu.add_text(axes[1, 1], 0.05, 0.95, "(b-2)", ha="left", va="top")
mu.add_text(axes[1, 2], 0.05, 0.95, "(b-3)", ha="left", va="top")

axes[0, 0].set_title("$f/f_{ce}=0.05$")
axes[0, 1].set_title("$f/f_{ce}=0.1$")
axes[0, 2].set_title("$f/f_{ce}=0.15$")
for i, j in np.ndindex(axes.shape):
    ax = axes[i, j]
    ax.locator_params(axis="x", nbins=5)
    ax.set_facecolor("silver")
    ax.set_yscale("log")
    ax.set_ylim(50, 2e3)
    ax.set_xlim(0, 180)
    if j == 0:
        ax.set_ylabel("$W$ (eV)")
    if i == 1:
        ax.set_xlabel("$\\alpha$ (deg)")

fig.tight_layout(w_pad=0.1, h_pad=0.05)
fig.savefig(work_dir / "plots" / "fig_weighted_energy_space.png", dpi=600)
mu.plt.show()
