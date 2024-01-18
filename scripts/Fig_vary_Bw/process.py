import config as cf
import numpy as np
import zarr
from rls.formula.conversions import cartesian_to_FAC
from rls.io import data_store


def process(ib):
    cf.sim.name = cf.name(ib)
    cf.model.Bw = np.abs(cf.Bw_B0[ib] * cf.units.B_factor)
    cf.sim.load_data()
    t = cf.sim.solutions.t[:]
    g, x, y, z, ux, uy, uz = cf.sim.solutions.as_tuple()
    tg = t[:, np.newaxis] * np.ones_like(x)

    _, _, _, B0_x, _, B0_z = cf.model.background_field(
        tg, x, y, z, *cf.model.background_field_args
    )
    wce = np.abs(
        cf.species.q.code / cf.species.m.code * np.sqrt(B0_x**2 + B0_z**2)
    )
    w, k = cf.model.dispersion_relation(wce, cf.w_wce, cf.wpe0.code, cf.c.code)

    # Wave-frame energization
    _, _, W, _ = cartesian_to_FAC(
        g, ux, uy, uz + w / k, B0_x, B0_z, cf.model
    )
    dW = (W[-1, :] - W[0, :]).user

    # Plasma-frame scattering
    V_para, V_perp, _, A = cartesian_to_FAC(
        g, ux, uy, uz, B0_x, B0_z, cf.model
    )
    Vi_para = V_para[0, :].user
    Vi_perp = V_perp[0, :].user
    dA = (A[-1, :] - A[0, :]).user

    resonant_particles = dA < 0
    kw = dict(
        x=Vi_para[resonant_particles],
        y=Vi_perp[resonant_particles],
        bins=bins,
    )
    H = np.histogram2d(**kw)[0]
    H_dW = np.histogram2d(weights=dW[resonant_particles], **kw)[0]
    H_dA = np.histogram2d(weights=np.degrees(dA[resonant_particles]), **kw)[0]
    H_dW2 = np.histogram2d(weights=dW[resonant_particles] ** 2, **kw)[0]
    H_dA2 = np.histogram2d(weights=dA[resonant_particles] ** 2, **kw)[0]
    dW = H_dW / H
    dA = H_dA / H
    dW2 = H_dW2 / H
    dA2 = H_dA2 / H

    print(f"Processed {ib}")
    return dW, dA, dW2, dA2


V_para_bins = np.linspace(-10, -2, 200) * cf.V_factor.user.unit
V_perp_bins = np.linspace(0, 10, 200) * cf.V_factor.user.unit
bins = (V_para_bins, V_perp_bins)
Bg, Vg_para, Vg_perp = np.meshgrid(
    cf.Bw_B0,
    V_para_bins[:-1],
    V_perp_bins[:-1],
    indexing="ij",
)
Wg_para = (0.5 * cf.species.m.user * Vg_para**2).to("eV")
Wg_perp = (0.5 * cf.species.m.user * Vg_perp**2).to("eV")

dW = np.zeros_like(Bg)
dA = np.zeros_like(Bg)
dW2 = np.zeros_like(Bg)
dA2 = np.zeros_like(Bg)
for ib in range(cf.Bw_B0.size):
    dW[ib, ...], dA[ib, ...], dW2[ib, ...], dA2[ib, ...] = process(ib)

zarr.save(
    store=data_store,
    path=cf.process_path,
    Bg=Bg,
    Vg_para=Vg_para,
    Vg_perp=Vg_perp,
    Wg_para=Wg_para,
    Wg_perp=Wg_perp,
    dW=dW,
    dA=dA,
    dW2=dW2,
    dA2=dA2,
)
