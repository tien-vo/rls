import config as cf
import numpy as np
import zarr
from labellines import labelLines
from scipy.ndimage import gaussian_filter
from tvolib import matplotlib_utils as mu

from rls.io import data_store

# ---- Process data for plots
Bg = zarr.open(store=data_store, path=cf.path("Bg"))[:]
Wg_perp = zarr.open(store=data_store, path=cf.path("Wg_perp"))[:]
Wg_para = zarr.open(store=data_store, path=cf.path("Wg_para"))[:]
dW = zarr.open(store=data_store, path=cf.path("dW"))[:]
dA = zarr.open(store=data_store, path=cf.path("dA"))[:]
Dpp = zarr.open(store=data_store, path=cf.path("Dpp"))[:]
Daa = zarr.open(store=data_store, path=cf.path("Daa"))[:]
Bg = Bg[:, :, 0]
Wg = Wg_para[:, :, 0]
Bw_B0 = Bg[:, 0]
W_arr = Wg[0, :]
dW_mean = np.nanmean(dW, axis=2)
dA_mean = np.nanmean(dA, axis=2)
Dp_mean = np.nanmean(Dpp, axis=2)
Da_mean = np.nanmean(Daa, axis=2)
dW_mean[np.isnan(dW_mean)] = 0.0
dA_mean[np.isnan(dA_mean)] = 0.0
start_wave_region = cf.calculate_resonant_energy(-1, s=-1)
end_wave_region = cf.calculate_resonant_energy(1, s=1)

# Eliminate non-physical haze
min_dA = np.nanmax(dA_mean)
for i, j in np.ndindex(dA_mean.shape):
    if (Wg[i, j] > start_wave_region[i]) & (Bg[i, j] > cf.R_lim):
        dA_mean[i, j] = min_dA

# ---- Create figure
fig = mu.plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(
    2, 3, width_ratios=(1.0, 0.2, 1.0), height_ratios=(1.0, 0.4)
)
slices = [70]
slices_color = ["k"]
tkw = dict(x=0.05, ha="left", va="top")

# -- Panel a (Energization)
ax_a = fig.add_subplot(gs[0, 0])
cax_a = mu.add_colorbar(ax_a)

im = ax_a.pcolormesh(Bg, Wg, gaussian_filter(dW_mean, 3), cmap="cet_rainbow4")
cb_a = fig.colorbar(im, cax=cax_a)
mu.add_panel_label(
    ax_a, text="(a) $\\langle\\Delta W\\rangle_\\perp$ (eV)", y=0.97, **tkw
)

# -- Panel c (Energy diffusion coefficient)
ax_c = fig.add_subplot(gs[1, 0])
mu.add_colorbar(ax_c).remove()

for i, iv in enumerate(slices):
    ax_c.scatter(
        cf.Bh_B0,
        Dp_mean[:, iv],
        fc=slices_color[i],
        ec="none",
        s=10,
        label=f"$W_{{\\|,0}}$={W_arr[iv]:.0f} eV",
    )

ax_c.set_ylabel("$\\langle\\Delta p^2\\rangle_\\perp/p^2$")
mu.add_panel_label(ax_c, text="(c)", y=0.9, **tkw)

# -- Panel b (Scattering)
ax_b = fig.add_subplot(gs[0, 2])
cax_b = mu.add_colorbar(ax_b)

im = ax_b.pcolormesh(
    Bg, Wg, gaussian_filter(dA_mean, 3), cmap="cet_rainbow4_r"
)
cb_b = fig.colorbar(im, cax=cax_b)
mu.add_panel_label(
    ax_b,
    text="(b) $\\langle\\Delta\\alpha\\rangle_\\perp$ (deg)",
    y=0.97,
    **tkw,
)

# -- Panel d (Pitch angle diffusion coefficient)
ax_d = fig.add_subplot(gs[1, 2])
mu.add_colorbar(ax_d).remove()

for i, iv in enumerate(slices):
    ax_d.scatter(
        cf.Bh_B0,
        Da_mean[:, iv],
        fc=slices_color[i],
        ec="none",
        s=10,
        label=f"$W_{{\\|,0}}$={W_arr[iv]:.0f} eV",
    )

ax_d.set_ylabel("$\\langle\\Delta\\alpha^2\\rangle_\\perp$")
mu.add_panel_label(ax_d, text="(d)", y=0.9, **tkw)

# -- Final format
ax_a.set_title("Energization")
ax_b.set_title("Scattering")
for ax in [ax_a, ax_b]:
    ax.set_xscale("log")
    ax.set_xticklabels([])
    ax.set_xlim(cf.Bh_B0[0], cf.Bh_B0[-1])
    ax.set_ylim(40, 280)
    ax.set_ylabel("$W_{\\|,0}$ (eV)")
    ax.axvline(cf.R_lim, ls="--", c="w", lw=1.5)
    ax.plot(cf.Bh_B0, start_wave_region, "-w", lw=1.5, label="$z=-R$")
    ax.plot(cf.Bh_B0, end_wave_region, "--w", lw=1.5, label="$z=R$")
    ax.tick_params(colors="w", labelcolor="k")
    for i, iv in enumerate(slices):
        ax.scatter(
            [Bw_B0[0]],
            [W_arr[iv]],
            ec="w",
            fc=slices_color[i],
            marker="X",
            s=200,
            zorder=999,
            clip_on=False,
        )

for ax in [ax_c, ax_d]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$B_h/B_0$")
    ax.set_xlim(cf.Bh_B0[0], cf.Bh_B0[-1])
    ax.axvline(cf.R_lim, ls="--", c="k", lw=1.5)

cmap = mu.mpl.colormaps.get_cmap("cet_rainbow4")
kw = dict(color="k", fontsize=16, outline_width=4, xvals=[3e-2, 3e-2])
labelLines(ax_a.get_lines(), **kw)
labelLines(ax_b.get_lines(), **kw)

ax_c.text(1.2e-2, 4e-4, "non-resonant", c="r")
ax_c.text(2.5e-1, 5e-3, "NL", c="r")
ax_c.text(1.1 * cf.R_lim, 1e-4, "QL", c="r")
ax_d.text(1.2e-2, 1e0, "non-resonant", c="r")
ax_d.text(3e-1, 3e1, "NL", c="r")
ax_d.text(1.1 * cf.R_lim, 7e-1, "QL", c="r")

ax_c.text(0.9 * cf.R_lim, 3e-5, "$(kR)(B_w/B_0)$", ha="right")
ax_d.text(0.9 * cf.R_lim, 2e-1, "$(kR)(B_w/B_0)$", ha="right")

fig.align_ylabels([ax_a, ax_c])
fig.align_ylabels([ax_b, ax_d])
fig.subplots_adjust(hspace=0.05, top=0.96, bottom=0.1, left=0.1, right=0.93)
fig.savefig(
    "../manuscript/assets/vary_gradient_hires.png",
    dpi=300,
)
fig.savefig(
    "../manuscript/assets/vary_gradient_lores.png",
    dpi=100,
)
mu.plt.show()
