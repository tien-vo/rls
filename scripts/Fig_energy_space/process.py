import config as cf
import numpy as np
import zarr
from rls.formula.conversions import cartesian_to_FAC
from rls.io import data_dir


def process(i):
    cf.sim.name = name[i]
    cf.model.w_wce = w_wce[i]

    cf.sim.load_data()
    t = cf.sim.solutions.t[:]
    g, x, y, z, ux, uy, uz = cf.sim.solutions.as_tuple()
    tg = t[:, np.newaxis] * np.ones_like(x)

    _, _, _, B0_x, _, B0_z = cf.model.background_field(
        tg, x, y, z, *cf.model.background_field_args
    )
    V_para, V_perp, W, A = cartesian_to_FAC(
        g, ux, uy, uz, B0_x, B0_z, cf.model
    )
    Iw = zarr.load(data_dir / f"{cf.sim.name}/raw_data/Iw")
    print(f"{Iw[Iw > 1.0].size / Iw.size:%} particles finished interacting")

    where = lambda var: data_dir / f"{cf.sim.name}/processed/{var}"
    zarr.save(where("Vi_para"), V_para[0, :].user)
    zarr.save(where("Vi_perp"), V_perp[0, :].user)
    zarr.save(where("Vf_para"), V_para[-1, :].user)
    zarr.save(where("Vf_perp"), V_perp[-1, :].user)
    zarr.save(where("Wi"), W[0, :].user)
    zarr.save(where("Wf"), W[-1, :].user)
    zarr.save(where("Ai"), A[0, :].user)
    zarr.save(where("Af"), A[-1, :].user)


name = [f"fig_energy_space_Bh_08_Bw_001_{x}" for x in ["A", "B", "C"]]
w_wce = [0.05, 0.1, 0.15]
process(0)
process(1)
process(2)
