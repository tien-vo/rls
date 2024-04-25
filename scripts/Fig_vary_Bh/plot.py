import colorcet
import config as cf
import numpy as np
import zarr
from astropy.units import Quantity
from rls.io import data_store, work_dir
from scipy.ndimage import gaussian_filter
from tvolib import mpl_utils as mu


def calculate_resonant_energy(z_R, V_perp=Quantity(2_000, "km/s"), s=0):
    species = cf.units.electron
    V_perp = (V_perp / cf.c.user).decompose().value * cf.c.code

    Wr = np.zeros_like(cf.Bh_B0)
    for ih in range(cf.Bh_B0.size):
        cf.model.Bh = -cf.Bh_B0[ih] * cf.units.B_factor
        B0_z = cf.B0 * cf.model.eta(
            z_R * cf.R.code,
            *cf.model.background_field_args
        )
        wce = np.abs(species.wc(B0_z)).code
        Vr = cf.model.resonant_velocity(
            wce, cf.w_wce, cf.wpe0.code, cf.c.code
        )
        dV = cf.model.resonant_width(
            V_perp, cf.Bw.code, B0_z.code, wce, cf.w_wce, cf.wpe0.code, cf.c.code
        )
        Wr[ih] = cf.W_factor.user.value * (0.5 * species.m.code * (Vr + s * dV)**2)

    return Wr


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
slices = [65]
slices_color = ["k"]

fig = mu.plt.figure(figsize=(12, 7))
tkw = dict(x=0.05, ha="left", va="top")
gs = fig.add_gridspec(
    2, 3, width_ratios=(1.0, 0.2, 1.0), height_ratios=(1.0, 0.4)
)

ax_a = fig.add_subplot(gs[0, 0])
cax = mu.add_colorbar(ax_a)
im = ax_a.pcolormesh(Bg, Wg, gaussian_filter(dW_mean, 3), cmap="cet_rainbow4")
cb = fig.colorbar(im, cax=cax)
mu.add_text(
    ax_a, text="(a) $\\langle\\Delta W\\rangle_\\perp$ (eV)", y=0.97, **tkw
)

ax_c = fig.add_subplot(gs[1, 0])
mu.add_colorbar(ax_c).remove()
ax_c.set_ylabel("$\\langle\\Delta p^2\\rangle_\\perp/p^2$")
mu.add_text(ax_c, text="(c)", y=0.9, **tkw)
for i, iv in enumerate(slices):
    ax_c.scatter(
        cf.Bh_B0,
        Dp_mean[:, iv],
        fc=slices_color[i],
        ec="none",
        s=10,
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
ax_d.set_ylabel("$\\langle\\Delta\\alpha^2\\rangle_\\perp$")
mu.add_text(ax_d, text="(d)", y=0.9, **tkw)
for i, iv in enumerate(slices):
    ax_d.scatter(
        cf.Bh_B0,
        Da_mean[:, iv],
        fc=slices_color[i],
        ec="none",
        s=10,
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
    ax.axvline(cf.R_lim, ls="--", c="w", lw=1.5)
    ax.fill_between(
        cf.Bh_B0,
        calculate_resonant_energy(-1, s=-1),
        calculate_resonant_energy(1, s=1),
        color="w",
        alpha=0.2,
    )
    for i, iv in enumerate(slices):
        ax.axhline(W_arr[iv], ls="--", c="w", lw=1.5)

for ax in [ax_c, ax_d]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$B_h/B_0$")
    ax.set_xlim(cf.Bh_B0[0], cf.Bh_B0[-1])
    ax.axvline(cf.R_lim, ls="--", c="k", lw=1.5)

ax_c.text(0.9 * cf.R_lim, 4e-5, "$(kR)(B_w/B_0)$", ha="right")
ax_d.text(0.9 * cf.R_lim, 4e-1, "$(kR)(B_w/B_0)$", ha="right")

fig.align_ylabels([ax_a, ax_c])
fig.align_ylabels([ax_b, ax_d])
fig.subplots_adjust(hspace=0.05, top=0.96, bottom=0.1, left=0.1, right=0.93)
fig.savefig(work_dir / "plots" / "fig_vary_Bh.png", dpi=600)
fig.savefig(work_dir / "plots" / "fig_vary_Bh_lowres.png")
#  mu.plt.show()
