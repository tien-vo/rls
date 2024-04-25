import config as cf
import numpy as np
import zarr
import click
from rls.formula.conversions import cartesian_to_FAC
from rls.io import data_dir


@click.command()
@click.option("--w-wce", default=0.05, help="Frequency")
@click.option("--Bw-B0", default=0.01, help="Magnitude")
@click.option("--Bh-B0", default=0.5, help="Depletion level")
def process(w_wce, bw_b0, bh_b0):
    cf.sim.name = (
        f"fig_energy_space_"
        f"w_{100 * w_wce:03.0f}_"
        f"Bw_{100 * bw_b0:03.0f}_"
        f"Bh_{10 * bh_b0:02.0f}"
    )
    cf.model.w_wce = w_wce
    cf.model.Bw = np.abs(bw_b0 * cf.units.B_factor)
    cf.model.Bh = -bh_b0 * cf.units.B_factor

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

if __name__ == "__main__":
    process()
