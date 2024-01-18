import numpy as np
from astropy.units import Quantity
from rls.data_types import Particle, SimQuantity
from rls.formula.conversions import energy_to_speed
from rls.formula.physics import lorentz_factor
from rls.models import WhistlerAtHyperbolicGradientModel
from rls.simulation import Simulation


def calculate_V(W, W_factor, units):
    _W = SimQuantity(W_factor.user_to_code(W), W)
    return energy_to_speed(_W, units)


sim = Simulation(
    name="fig_setup_random_IC",
    model=WhistlerAtHyperbolicGradientModel(
        Bw=0.05,
        w_wce=0.1,
        sw=1.75,
        Bh=-0.8,
        B0=-1.0,
        theta=0.0,
    ),
)
model = sim.model
units = model.units
species = units.electron
W_factor = units.W_factor
c = model.units.c
R = model.R
Tc = species.cyclotron_period(model.B0)

rng = np.random.default_rng()
number_of_particles = 1_000_000
V_para_min = calculate_V(Quantity(10.0, "eV"), W_factor, units).code
V_para_max = calculate_V(Quantity(400.0, "eV"), W_factor, units).code
V_perp_min = calculate_V(Quantity(0.0, "eV"), W_factor, units).code
V_perp_max = calculate_V(Quantity(300.0, "eV"), W_factor, units).code
V_para = rng.uniform(V_para_min, V_para_max, number_of_particles)
V_perp = rng.uniform(V_perp_min, V_perp_max, number_of_particles)
phase = rng.uniform(-np.pi, np.pi, number_of_particles)
uz0 = V_para
ux0 = V_perp * np.cos(phase)
uy0 = V_perp * np.sin(phase)
x0 = np.zeros_like(uz0)
y0 = np.zeros_like(uz0)
z0 = np.zeros_like(uz0) - 2 * R.code
g0 = lorentz_factor(ux0, uy0, uz0, c.code)
particles = Particle(species, 0.0, g0, x0, y0, z0, ux0, uy0, uz0)

W_run = Quantity(10.0, "eV")
W_run = SimQuantity(W_factor.user_to_code(W_run), W_run)
V_run = energy_to_speed(W_run, units)
T_run = (4 * R / V_run).code
sim.run(
    initial_conditions=particles,
    run_time=T_run,
    step_size=1e-2 * Tc.code,
    save_intervals=3,
    log_intervals=1000,
)
sim.save_data()
