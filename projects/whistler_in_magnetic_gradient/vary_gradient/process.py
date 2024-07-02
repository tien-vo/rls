import config as cf
import numpy as np
import zarr
from rls.formula.conversions import cartesian_to_FAC
from rls.io import data_store


def process(ih):
    cf.sim.name = cf.name(ih)
    cf.model.Bh = -cf.Bh_B0[ih] * cf.units.B_factor
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

    V_para, V_perp, W, A = cartesian_to_FAC(
        g, ux, uy, uz, B0_x, B0_z, cf.model
    )
    V_para = V_para.user
    V_perp = V_perp.user
    W = W.user
    A = np.degrees(A.user)
    Vi_para = V_para[0, :]
    Vi_perp = V_perp[0, :]
    Wi = W[0, :][None, :]
    Ai = A[0, :][None, :]
    dW = np.nanmean(W - Wi, axis=0)
    dA = np.nanmean(A - Ai, axis=0)
    V = np.sqrt(V_para**2 + V_perp**2)
    Dpp = np.nanmean(
        ((V_para - Vi_para[None, :])**2
        + (V_perp - Vi_perp[None, :])**2) / V**2, axis=0)
    Daa = np.nanmean((A - Ai)**2, axis=0)

    resonant_particles = dA < 0
    kw = dict(
        x=Vi_para[resonant_particles],
        y=Vi_perp[resonant_particles],
        bins=bins,
    )
    H = np.histogram2d(**kw)[0]
    H_dW = np.histogram2d(weights=dW[resonant_particles], **kw)[0]
    H_dA = np.histogram2d(weights=dA[resonant_particles], **kw)[0]
    H_Dpp = np.histogram2d(weights=Dpp[resonant_particles], **kw)[0]
    H_Daa = np.histogram2d(weights=Daa[resonant_particles], **kw)[0]
    dW = H_dW / H
    dA = H_dA / H
    Dpp = H_Dpp / H
    Daa = H_Daa / H

    print(f"Processed {ib}")
    return dW, dA, Dpp, Daa


V_para_bins = np.linspace(-10, -2, 200) * cf.V_factor.user.unit
V_perp_bins = np.linspace(0, 10, 200) * cf.V_factor.user.unit
bins = (V_para_bins, V_perp_bins)
Bg, Vg_para, Vg_perp = np.meshgrid(
    cf.Bh_B0,
    V_para_bins[:-1],
    V_perp_bins[:-1],
    indexing="ij",
)
Wg_para = (0.5 * cf.species.m.user * Vg_para**2).to("eV")
Wg_perp = (0.5 * cf.species.m.user * Vg_perp**2).to("eV")

dW = np.zeros_like(Bg)
dA = np.zeros_like(Bg)
Dpp = np.zeros_like(Bg)
Daa = np.zeros_like(Bg)
for ib in range(cf.Bh_B0.size):
    dW[ib, ...], dA[ib, ...], Dpp[ib, ...], Daa[ib, ...] = process(ib)

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
    Dpp=Dpp,
    Daa=Daa,
)
