import colorcet
import config as cf
import numpy as np
import zarr
from rls.io import data_store, work_dir
from tvolib import mpl_utils as mu


def calculate_resonant_energy(z_R):
    species = cf.units.electron
    B0_z = cf.B0 * cf.model.eta(
        z_R * cf.R.code,
        *cf.model.background_field_args
    )
    wce = np.abs(species.wc(B0_z)).code
    Vr = cf.model.resonant_velocity(
        wce, cf.w_wce, cf.wpe0.code, cf.c.code
    )
    return cf.W_factor.user * (0.5 * species.m.code * Vr**2)


path = lambda x: f"{cf.process_path}/{x}"
Bg = zarr.open(store=data_store, path=path("Bg"))[:]
Wg_perp = zarr.open(store=data_store, path=path("Wg_perp"))[:]
Wg_para = zarr.open(store=data_store, path=path("Wg_para"))[:]
dW = zarr.open(store=data_store, path=path("dW"))[:]
dA = zarr.open(store=data_store, path=path("dA"))[:]
Dpp = zarr.open(store=data_store, path=path("Dpp"))[:]
Daa = zarr.open(store=data_store, path=path("Daa"))[:]
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
slices = [75, 55, 35]
slices_color = ["g", "r", "b"]

fig = mu.plt.figure(figsize=(12, 7))
tkw = dict(x=0.05, ha="left", va="top")
gs = fig.add_gridspec(
    2, 3, width_ratios=(1.0, 0.2, 1.0), height_ratios=(1.0, 0.4)
)

ax_a = fig.add_subplot(gs[0, 0])
cax = mu.add_colorbar(ax_a)
im = ax_a.pcolormesh(Bg, Wg, dW_mean, cmap="cet_rainbow4")
fig.colorbar(im, cax=cax)
mu.add_text(
    ax_a, text="(a) $\\langle\\Delta W\\rangle_\\perp$ (eV)", y=0.97, **tkw
)

ax_c = fig.add_subplot(gs[1, 0])
mu.add_colorbar(ax_c).remove()
B_arr = Bw_B0[Bw_B0 <= 1e-3]
ax_c.plot(B_arr, y := (4e3 * B_arr**2), "--k")
ax_c.text(0.5 * B_arr[-1], 2 * y[-1], "$B_w^2$")
B_arr = Bw_B0[Bw_B0 >= 1e-1]
ax_c.plot(B_arr, y := (8e-2 * B_arr ** (0.5)), "--k")
ax_c.text(1.2 * B_arr[0], 1e-1 * y[0], "$B_w^{1/2}$")
ax_c.set_ylabel("$\\langle\\Delta p^2\\rangle_\\perp/p^2$")
mu.add_text(ax_c, text="(c)", y=0.9, **tkw)
for i, iv in enumerate(slices):
    ax_c.scatter(
        Bw_B0,
        Dp_mean[:, iv],
        fc=slices_color[i],
        ec="none",
        s=10,
    )

ax_b = fig.add_subplot(gs[0, 2])
cax = mu.add_colorbar(ax_b)
im = ax_b.pcolormesh(Bg, Wg, dA_mean, cmap="cet_rainbow4_r")
fig.colorbar(im, cax=cax)
mu.add_text(
    ax_b,
    text="(b) $\\langle\\Delta\\alpha\\rangle_\\perp$ (deg)",
    y=0.97,
    **tkw,
)

ax_d = fig.add_subplot(gs[1, 2])
mu.add_colorbar(ax_d).remove()
B_arr = Bw_B0[Bw_B0 <= 1e-3]
ax_d.plot(B_arr, y := (2e7 * B_arr**2), "--k")
ax_d.text(0.5 * B_arr[-1], 2 * y[-1], "$B_w^2$")
B_arr = Bw_B0[Bw_B0 >= 1e-1]
ax_d.plot(B_arr, y := (2e2 * B_arr ** (0.5)), "--k")
ax_d.text(1.2 * B_arr[0], 1e-1 * y[0], "$B_w^{1/2}$")
ax_d.set_ylabel("$\\langle\\Delta\\alpha^2\\rangle_\\perp$")
mu.add_text(ax_d, text="(d)", y=0.9, **tkw)
for i, iv in enumerate(slices):
    ax_d.scatter(
        Bw_B0,
        Da_mean[:, iv],
        fc=slices_color[i],
        ec="none",
        s=10,
    )


ax_a.set_title("Energization")
ax_b.set_title("Scattering")
for ax in [ax_a, ax_b]:
    ax.set_xscale("log")
    ax.set_xticklabels([])
    ax.set_xlim(Bw_B0[0], Bw_B0[-1])
    ax.set_ylim(40, 280)
    ax.set_ylabel("$W_{\\|,0}$ (eV)")
    ax.axvline(cf.R_lim, ls="--", c="w", lw=1.5)
    ax.axhspan(
        calculate_resonant_energy(-1).value,
        calculate_resonant_energy(1).value,
        color="w",
        alpha=0.2,
    )
    for i, iv in enumerate(slices):
        ax.axhline(W_arr[iv], ls="--", c=slices_color[i], lw=1.5)

for ax in [ax_c, ax_d]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$B_w/B_0$")
    ax.set_xlim(Bw_B0[0], Bw_B0[-1])
    ax.axvline(cf.R_lim, ls="--", c="k", lw=1.5)

ax_c.set_yticks(10.0 ** np.arange(-7, 1, 2))
ax_c.set_ylim(1e-7, 1e0)
ax_c.text(1.2 * cf.R_lim, 4e-7, "$(B_h/B_0)/(kR)$")
ax_d.set_yticks(10.0 ** np.arange(-3, 5, 2))
ax_d.set_ylim(1e-3, 1e4)
ax_d.text(1.2 * cf.R_lim, 4e-3, "$(B_h/B_0)/(kR)$")

fig.align_ylabels([ax_a, ax_c])
fig.align_ylabels([ax_b, ax_d])
fig.subplots_adjust(hspace=0.05, top=0.96, bottom=0.1, left=0.1, right=0.94)
fig.savefig(work_dir / "plots" / "fig_vary_Bw.png", dpi=600)
fig.savefig(work_dir / "plots" / "fig_vary_Bw_lowres.png")
# mu.plt.show()
