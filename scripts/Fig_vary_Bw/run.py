import click
import config as cf
import numpy as np
from astropy.units import Quantity
from rls.data_types import Particle, SimQuantity
from rls.formula.conversions import energy_to_speed
from rls.formula.physics import lorentz_factor


@click.command()
@click.option("--ib", default=0, help="Wave amplitude index")
def run(ib):
    cf.sim.name = cf.name(ib)
    cf.model.Bw = np.abs(cf.Bw_B0[ib] * cf.units.B_factor)

    number_of_particles = 500_000
    W_min = Quantity(0.0, "eV")
    W_max = Quantity(1000.0, "eV")
    W_min = SimQuantity(cf.W_factor.user_to_code(W_min), W_min)
    W_max = SimQuantity(cf.W_factor.user_to_code(W_max), W_max)
    V_min = energy_to_speed(W_min, cf.units)
    V_max = energy_to_speed(W_max, cf.units)

    rng = np.random.default_rng()
    V_para = rng.uniform(V_min.code, V_max.code, number_of_particles)
    V_perp = rng.uniform(V_min.code, V_max.code, number_of_particles)
    phase = rng.uniform(-np.pi, np.pi, number_of_particles)
    uz0 = V_para
    ux0 = V_perp * np.cos(phase)
    uy0 = V_perp * np.sin(phase)
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
        save_intervals=20,
        log_intervals=1000,
    )
    cf.sim.save_data()


if __name__ == "__main__":
    run()
