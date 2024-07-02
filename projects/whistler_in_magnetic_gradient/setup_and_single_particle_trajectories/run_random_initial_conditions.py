import config as cf
import numpy as np
from astropy.units import Quantity

from rls.data_types import Particle
from rls.formula.physics import lorentz_factor

rng = np.random.default_rng()
number_of_particles = 1_000_000
V_para_min = cf.get_simulation_speed(Quantity(10.0, "eV")).code
V_para_max = cf.get_simulation_speed(Quantity(400.0, "eV")).code
V_perp_min = cf.get_simulation_speed(Quantity(0.0, "eV")).code
V_perp_max = cf.get_simulation_speed(Quantity(300.0, "eV")).code
V_para = rng.uniform(V_para_min, V_para_max, number_of_particles)
V_perp = rng.uniform(V_perp_min, V_perp_max, number_of_particles)
phase = rng.uniform(-np.pi, np.pi, number_of_particles)
uz0 = V_para
ux0 = V_perp * np.cos(phase)
uy0 = V_perp * np.sin(phase)
x0 = np.zeros_like(uz0)
y0 = np.zeros_like(uz0)
z0 = np.zeros_like(uz0) - 2 * cf.model.R.code
g0 = lorentz_factor(ux0, uy0, uz0, cf.units.c.code)
initial_conditions = Particle(cf.electron, 0.0, g0, x0, y0, z0, ux0, uy0, uz0)

V_run = cf.get_simulation_speed(Quantity(10.0, "eV"))
T_run = (4 * cf.model.R / V_run).code

cf.sim.name = "random_initial_conditions"
cf.sim.run(
    initial_conditions=initial_conditions,
    run_time=T_run,
    step_size=1e-2 * cf.Tce0.code,
    save_intervals=3,
    log_intervals=1000,
)
cf.sim.save_data()
