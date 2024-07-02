import config as cf
import numpy as np
import zarr
from labellines import labelLines
from tvolib import matplotlib_utils as mu

from rls.io import data_store, work_dir

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

# ---- Create figure
fig = mu.plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(
    4, 3, width_ratios=(1.0, 0.2, 1.0), height_ratios=(1.0, 0.4, 0.2, 0.4)
)

tkw = dict(x=0.03, ha="left", va="top")
slices = [75, 55, 35]
slices_color = ["g", "r", "b"]

# -- Panel a (Energization)
ax_a = fig.add_subplot(gs[0, 0])
cax_a = mu.add_colorbar(ax_a)

im = ax_a.pcolormesh(Bg, Wg, dW_mean, cmap="cet_rainbow4")
fig.colorbar(im, cax=cax_a)
mu.add_panel_label(
    ax_a, text="(a) $\\langle\\Delta W\\rangle_\\perp$ (eV)", y=0.97, **tkw
)

# -- Panel c (Energy diffusion coefficient)
ax_c = fig.add_subplot(gs[1, 0])
mu.add_colorbar(ax_c).remove()

for i, iv in enumerate(slices):
    ax_c.scatter(Bw_B0, Dp_mean[:, iv], fc=slices_color[i], ec="none", s=10)

B_arr = Bw_B0[Bw_B0 <= 1e-3]
ax_c.plot(B_arr, y := (4e3 * B_arr**2), "--k")
ax_c.text(0.5 * B_arr[-1], 2 * y[-1], "$B_w^2$")
B_arr = Bw_B0[Bw_B0 >= 1e-1]
ax_c.plot(B_arr, y := (8e-2 * B_arr ** (0.5)), "--k")
ax_c.text(1.2 * B_arr[0], 1e-1 * y[0], "$B_w^{1/2}$")

ax_c.set_ylabel("$\\langle\\Delta p^2\\rangle_\\perp/p^2$")
mu.add_panel_label(ax_c, text="(c)", y=0.92, **tkw)

# -- Panel b (Scattering)
ax_b = fig.add_subplot(gs[0, 2])
cax_b = mu.add_colorbar(ax_b)

im = ax_b.pcolormesh(Bg, Wg, dA_mean, cmap="cet_rainbow4_r")
fig.colorbar(im, cax=cax_b)
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
    ax_d.scatter(Bw_B0, Da_mean[:, iv], fc=slices_color[i], ec="none", s=10)

B_arr = Bw_B0[Bw_B0 <= 1e-3]
ax_d.plot(B_arr, y := (2e7 * B_arr**2), "--k")
ax_d.text(0.5 * B_arr[-1], 2 * y[-1], "$B_w^2$")
B_arr = Bw_B0[Bw_B0 >= 1e-1]
ax_d.plot(B_arr, y := (3e2 * B_arr ** (0.5)), "--k")
ax_d.text(1.2 * B_arr[0], 1e-1 * y[0], "$B_w^{1/2}$")

ax_d.set_ylabel("$\\langle\\Delta\\alpha^2\\rangle_\\perp$")
mu.add_panel_label(ax_d, text="(d)", y=0.92, **tkw)


# -- Panel e (Resonance broadening mechanisms)
ax_e = fig.add_subplot(gs[3, :])
mu.add_colorbar(ax_e, size="0.7%").remove()

V_perp_arr = np.linspace(0, 0.08, 1000) * cf.c
V_start, V_end, Vc_start, Vc_end = None, None, None, None
for iz, z in enumerate(np.linspace(-1, 1, Nz := 5)):
    Vr, Vp, Vm = cf.calculate_resonant_velocity(V_perp_arr.code, 1e-3, 0.3, z)
    Vr = Vr * cf.V_factor.user.to("1000 km/s").value
    Vp = Vp * cf.V_factor.user.to("1000 km/s").value
    Vm = Vm * cf.V_factor.user.to("1000 km/s").value

    kw = dict(c="k")
    if iz in [0, Nz - 1]:
        ax_e.text(
            Vr,
            3.8,
            f"$z/R={z.astype(np.int64)}$",
            color="k",
            fontsize="large",
            ha="center",
            va="top",
        )
        if iz == 0:
            V_start = Vm
            Vc_start = Vr

            ax_e.plot(
                Vm, V_perp_arr.user.to("1000 km/s").value, ls="-", lw=3, **kw
            )
            ax_e.plot(
                Vp, V_perp_arr.user.to("1000 km/s").value, ls=":", lw=1, **kw
            )
        else:
            V_end = Vp
            Vc_end = Vr

            ax_e.plot(
                Vm, V_perp_arr.user.to("1000 km/s").value, ls=":", lw=1, **kw
            )
            ax_e.plot(
                Vp, V_perp_arr.user.to("1000 km/s").value, ls="--", lw=3, **kw
            )
    else:
        ax_e.plot(
            Vm, V_perp_arr.user.to("1000 km/s").value, ls=":", lw=1, **kw
        )
        ax_e.plot(
            Vp, V_perp_arr.user.to("1000 km/s").value, ls=":", lw=1, **kw
        )

ax_e.fill_betweenx(
    V_perp_arr.user.to("1000 km/s").value, V_start, V_end, color="k", alpha=0.2
)
ax_e.annotate(
    text="",
    xy=(-7.25, 2.7),
    xytext=(-6.5, 2.7),
    arrowprops=dict(arrowstyle="<->", linewidth=2),
)
ax_e.annotate(
    text="",
    xy=(Vc_start, 0.5),
    xytext=(Vc_end, 0.5),
    arrowprops=dict(arrowstyle="<->", linewidth=2),
)
ax_e.text(-6.9, 0.85, "$\\Delta V_g$", ha="center", fontsize="xx-large")
ax_e.text(
    -6.9, 3.7, "$\\Delta V_w$", ha="center", va="top", fontsize="xx-large"
)

ax_e.set_xlabel("$V_\\|$ (1000 km/s)")
ax_e.set_ylabel("$V_\\perp$ (1000 km/s)")
ax_e.set_xlim(-8.7, -5.5)
ax_e.set_ylim(0, 4)
ax_e.set_xticks(np.arange(-8, -5, 1))
ax_e.set_yticks(np.arange(0, 5, 1))
ax_e.text(-8.6, 0.5, "$B_w/B_0=10^{-3}$", fontsize="large")
ax_e.text(-8.6, 1.5, "$B_h/B_0=0.3$", fontsize="large")
mu.add_panel_label(ax_e, text="(e)", x=0.015, y=0.9, va="top")

# -- Final format
ax_a.set_title("Energization")
ax_b.set_title("Scattering")
for ax in [ax_a, ax_b]:
    ax.set_xscale("log")
    ax.set_xticklabels([])
    ax.set_xlim(Bw_B0[0], Bw_B0[-1])
    ax.set_ylim(40, 280)
    ax.set_ylabel("$W_{\\|,0}$ (eV)")
    ax.axvline(cf.R_lim, ls="--", c="w", lw=1.5)
    ax.plot(Bw_B0, start_wave_region, c="w", lw=1.5, label="$z=-R$")
    ax.plot(Bw_B0, end_wave_region, c="w", ls="--", lw=1.5, label="$z=R$")
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

cmap = mu.mpl.colormaps.get_cmap("cet_rainbow4")
kw = dict(color="k", fontsize=16, outline_width=4, xvals=[5e-4, 5e-4])  # type: ignore
labelLines(ax_a.get_lines(), **kw)
labelLines(ax_b.get_lines(), **kw)

for ax in [ax_c, ax_d]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$B_w/B_0$")
    ax.set_xlim(Bw_B0[0], Bw_B0[-1])
    ax.axvline(cf.R_lim, ls="--", c="k", lw=1.5)

ax_a.text(0.8 * cf.R_lim, 110, "QL", ha="right", c="w")
ax_a.text(1.2 * cf.R_lim, 110, "NL", ha="left", c="w")
ax_b.text(0.8 * cf.R_lim, 110, "QL", ha="right", c="w")
ax_b.text(1.2 * cf.R_lim, 110, "NL", ha="left", c="w")

ax_c.set_yticks(10.0 ** np.arange(-7, 1, 2))
ax_c.set_ylim(1e-7, 1e0)
ax_c.text(1.2 * cf.R_lim, 4e-7, "$(B_h/B_0)/(kR)$")
ax_d.set_yticks(10.0 ** np.arange(-3, 5, 2))
ax_d.set_ylim(1e-3, 1e4)
ax_d.text(1.2 * cf.R_lim, 4e-3, "$(B_h/B_0)/(kR)$")

fig.align_ylabels([ax_a, ax_c])
fig.align_ylabels([ax_b, ax_d])
fig.subplots_adjust(hspace=0.05, top=0.96, bottom=0.1, left=0.1, right=0.94)
fig.savefig(
    "../manuscript/assets/vary_wave_amplitude_hires.png",
    dpi=300,
)
fig.savefig(
    "../manuscript/assets/vary_wave_amplitude_lores.png",
    dpi=100,
)
mu.plt.show()
