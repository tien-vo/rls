import colorcet
import config as cf
import numpy as np
import zarr
from rls.io import data_store, work_dir
from scipy.ndimage import gaussian_filter
from tvolib import mpl_utils as mu

path = lambda x: f"{cf.process_path}/{x}"
Bg = zarr.open(store=data_store, path=path("Bg"))[:]
Wg_perp = zarr.open(store=data_store, path=path("Wg_perp"))[:]
Wg_para = zarr.open(store=data_store, path=path("Wg_para"))[:]
dW = zarr.open(store=data_store, path=path("dW"))[:]
dA = zarr.open(store=data_store, path=path("dA"))[:]
dW2 = zarr.open(store=data_store, path=path("dW2"))[:]
dA2 = zarr.open(store=data_store, path=path("dA2"))[:]
Bg = Bg[:, :, 0]
Wg = Wg_para[:, :, 0]
Bw_B0 = Bg[:, 0]
W_arr = Wg[0, :]
dW_mean = np.nanmean(dW, axis=2)
dA_mean = np.nanmean(dA, axis=2)
Dw_mean = np.nanmean(dW2, axis=2)
Da_mean = np.nanmean(dA2, axis=2)
dW_mean[np.isnan(dW_mean)] = 0.0
dA_mean[np.isnan(dA_mean)] = 0.0
slices = [
    60,
]
slices_color = [
    "k",
]

fig = mu.plt.figure(figsize=(12, 7))
tkw = dict(x=0.05, ha="left", va="top")
gs = fig.add_gridspec(
    2, 3, width_ratios=(1.0, 0.2, 1.0), height_ratios=(1.0, 0.25)
)

ax_a = fig.add_subplot(gs[0, 0])
cax = mu.add_colorbar(ax_a)
im = ax_a.pcolormesh(Bg, Wg, gaussian_filter(dW_mean, 3), cmap="cet_rainbow4")
cb = fig.colorbar(im, cax=cax)
cb.set_label("eV")
mu.add_text(
    ax_a, text="(a) $\\langle\\Delta W\\rangle_\\perp$ (eV)", y=0.97, **tkw
)

ax_c = fig.add_subplot(gs[1, 0])
mu.add_colorbar(ax_c).remove()
#  B_arr = cf.Bh_B0[cf.Bh_B0 >= 1e-1]
#  ax_c.plot(B_arr, y := (3e3 * B_arr**3), "--k")
#  ax_c.text(0.5 * B_arr[0], 0.5 * y[0], "$B_h^3$")
ax_c.set_ylabel("$\\langle\\Delta W^2\\rangle_\\perp$")
mu.add_text(ax_c, text="(c)", y=0.9, **tkw)
for i, iv in enumerate(slices):
    ax_c.scatter(
        cf.Bh_B0,
        Dw_mean[:, iv],
        ec=slices_color[i],
        fc="w",
        s=5,
        label=f"$W_{{\\|,0}}$={W_arr[iv]:.0f} eV",
    )

ax_b = fig.add_subplot(gs[0, 2])
cax = mu.add_colorbar(ax_b)
im = ax_b.pcolormesh(
    Bg, Wg, gaussian_filter(dA_mean, 3), cmap="cet_rainbow4_r"
)
cb_b = fig.colorbar(im, cax=cax)
mu.add_text(
    ax_b,
    text="(b) $\\langle\\Delta\\alpha\\rangle_\\perp$ (deg)",
    y=0.97,
    **tkw,
)

ax_d = fig.add_subplot(gs[1, 2])
mu.add_colorbar(ax_d).remove()
# B_arr = cf.Bh_B0[(5e-2 <= cf.Bh_B0) & (cf.Bh_B0 <= 5e-1)]
# ax_d.plot(B_arr, y := (5e0 * B_arr**2), "--k")
# ax_d.text(0.8 * B_arr[0], 0.8 * y[0], "$B_h^2$")
ax_d.set_ylabel("$\\langle\\Delta\\alpha^2\\rangle_\\perp$")
mu.add_text(ax_d, text="(d)", y=0.9, **tkw)
for i, iv in enumerate(slices):
    ax_d.scatter(
        cf.Bh_B0,
        Da_mean[:, iv],
        ec=slices_color[i],
        fc="w",
        s=5,
        label=f"$W_{{\\|,0}}$={W_arr[iv]:.0f} eV",
    )

ax_a.set_title("Energization")
ax_b.set_title("Scattering")
for ax in [ax_a, ax_b]:
    ax.set_xscale("log")
    ax.set_xticklabels([])
    ax.set_xlim(cf.Bh_B0[0], cf.Bh_B0[-1])
    ax.set_ylim(40, 280)
    ax.set_ylabel("$W_{\\|,0}$ (eV)")
    for i, iv in enumerate(slices):
        ax.axhline(W_arr[iv], ls="--", c="w")

for ax in [ax_c, ax_d]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$B_h/B_0$")
    ax.set_xlim(cf.Bh_B0[0], cf.Bh_B0[-1])

ax_c.set_ylim(1e-1, 1e4)
ax_d.set_ylim(1e-3, 1e0)

fig.align_ylabels([ax_a, ax_c])
fig.align_ylabels([ax_b, ax_d])
fig.subplots_adjust(hspace=0.05, top=0.96, bottom=0.1, left=0.1, right=0.93)
fig.savefig(work_dir / "plots" / "fig_vary_Bh.png", dpi=600)
#  mu.plt.show()
