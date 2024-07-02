import click
import config as cf
import zarr
from rls.io import data_dir
import numpy as np
from astropy.units import Quantity
from rls.data_types import Particle, SimQuantity
from rls.formula.conversions import energy_to_speed
from rls.formula.physics import lorentz_factor


def calculate_V(W, W_factor, units):
    _W = SimQuantity(cf.W_factor.user_to_code(W), W)
    return energy_to_speed(_W, units)


@click.command()
@click.option("--w-wce", default=0.05, help="Frequency")
@click.option("--Bw-B0", default=0.01, help="Magnitude")
@click.option("--Bh-B0", default=0.5, help="Depletion level")
def run(w_wce, bw_b0, bh_b0):
    cf.sim.name = (
        f"fig_energy_space_"
        f"w_{100 * w_wce:03.0f}_"
        f"Bw_{100 * bw_b0:03.0f}_"
        f"Bh_{10 * bh_b0:02.0f}"
    )
    cf.model.w_wce = w_wce
    cf.model.Bw = np.abs(bw_b0 * cf.units.B_factor)
    cf.model.Bh = -bh_b0 * cf.units.B_factor

    rng = np.random.default_rng()
    number_of_particles = 1_000_000
    W = 10.0 ** rng.uniform(np.log10(50), np.log10(2050), number_of_particles)
    A = rng.uniform(0.0, np.radians(180), number_of_particles)
    phase = rng.uniform(-np.pi, np.pi, number_of_particles)
    V = calculate_V(Quantity(W, "eV"), cf.W_factor, cf.units).code
    uz0 = V * np.cos(A)
    ux0 = V * np.sin(A) * np.cos(phase)
    uy0 = V * np.sin(A) * np.sin(phase)
    x0 = np.zeros_like(uz0)
    y0 = np.zeros_like(uz0)
    z0 = np.zeros_like(uz0) - 2 * cf.R.code
    g0 = lorentz_factor(ux0, uy0, uz0, cf.c.code)
    particles = Particle(cf.species, 0.0, g0, x0, y0, z0, ux0, uy0, uz0)

    W_run = Quantity(10.0, "eV")
    W_run = SimQuantity(cf.W_factor.user_to_code(W_run), W_run)
    V_run = energy_to_speed(W_run, cf.units)
    T_run = (4 * cf.R / V_run).code
    cf.sim.run(
        initial_conditions=particles,
        run_time=T_run,
        step_size=1e-2 * cf.Tc.code,
        save_intervals=3,
        log_intervals=1000,
    )
    cf.sim.save_data()
    zarr.save(data_dir / f"{cf.sim.name}/raw_data/Iw", cf.sim.Iw)

if __name__ == "__main__":
    run()
