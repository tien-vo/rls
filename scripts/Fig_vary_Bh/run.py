import click

import numpy as np
from astropy.units import Quantity

from rls.simulation import Simulation
from rls.data_types import Particle, SimQuantity
from rls.formula.physics import lorentz_factor
from rls.formula.conversions import energy_to_speed
from rls.models import WhistlerAtGradientModel


@click.command()
@click.option("--ih", default=0, help="Depletion level index")
def run(ih):
    sim = Simulation(
        name=f"vary_Bh_{ih}",
        model=WhistlerAtGradientModel(
            w_wce=0.05,
            sw=1.75,
            Bw=0.05,
            Bh=-Bh_B0[ih],
            B0=-1.0,
            theta=0.0,
        )
    )
    model = sim.model
    units = model.units
    species = units.scaling_species
    W_factor = units.W_factor
    c = model.units.c
    R = model.R
    Tc = species.cyclotron_period(model.B0)

    number_of_particles=500_000
    W_min=Quantity(0.0, "eV")
    W_max=Quantity(1000.0, "eV")
    W_min = SimQuantity(W_factor.user_to_code(W_min), W_min)
    W_max = SimQuantity(W_factor.user_to_code(W_max), W_max)
    V_min = energy_to_speed(W_min, units)
    V_max = energy_to_speed(W_max, units)

    rng = np.random.default_rng()
    V_para = rng.uniform(V_min.code, V_max.code, number_of_particles)
    V_perp = rng.uniform(V_min.code, V_max.code, number_of_particles)
    phase = rng.uniform(-np.pi, np.pi, number_of_particles)
    uz0 = V_para
    ux0 = V_perp * np.cos(phase)
    uy0 = V_perp * np.sin(phase)
    x0 = np.zeros_like(uz0)
    y0 = np.zeros_like(uz0)
    z0 = np.zeros_like(uz0) - 3 * R.code
    g0 = lorentz_factor(ux0, uy0, uz0, c.code)
    particles = Particle(species, 0.0, g0, x0, y0, z0, ux0, uy0, uz0)

    W_run = Quantity(10.0, "eV")
    W_run = SimQuantity(W_factor.user_to_code(W_run), W_run)
    V_run = energy_to_speed(W_run, units)
    T_run = (6 * R / V_run).code
    sim.run(
        initial_conditions=particles,
        run_time=T_run,
        step_size=1e-2 * Tc.code,
        save_intervals=3,
        log_intervals=100,
    )
    sim.save_data()


if __name__ == "__main__":
    Bh_B0 = np.linspace(0.0, 1.0, 50)
    run()
