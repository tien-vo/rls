import config as cf
import numpy as np
import zarr
from astropy.units import Quantity
from rls.data_types import Particle, SimQuantity
from rls.formula.conversions import energy_to_speed
from rls.formula.physics import lorentz_factor

cf.sim.name = "realistic_distribution"
cf.model.w_wce = 0.05
cf.model.Bw = np.abs(0.01 * cf.units.B_factor)
cf.model.Bh = -0.8 * cf.units.B_factor

rng = np.random.default_rng()
number_of_particles = 100_000
W_min = Quantity(0.0, "eV")
W_max = Quantity(2000.0, "eV")
W_min = SimQuantity(cf.W_factor.user_to_code(W_min), W_min)
W_max = SimQuantity(cf.W_factor.user_to_code(W_max), W_max)
V_min = energy_to_speed(W_min, cf.units)
V_max = energy_to_speed(W_max, cf.units)

rng = np.random.default_rng()
V_para = rng.uniform(-V_max.code, V_max.code, number_of_particles)
V_perp = rng.uniform(V_min.code, V_max.code, number_of_particles)
phase = rng.uniform(-np.pi, np.pi, number_of_particles)
uz0 = V_para
ux0 = V_perp * np.cos(phase)
uy0 = V_perp * np.sin(phase)
x0 = np.zeros_like(uz0)
y0 = np.zeros_like(uz0)
z0 = np.zeros_like(uz0) - 3 * cf.R.code
g0 = lorentz_factor(ux0, uy0, uz0, cf.c.code)
particles = Particle(cf.species, 0.0, g0, x0, y0, z0, ux0, uy0, uz0)

W_run = Quantity(10.0, "eV")
W_run = SimQuantity(cf.W_factor.user_to_code(W_run), W_run)
V_run = energy_to_speed(W_run, cf.units)
T_run = (6 * cf.R / V_run).code
cf.sim.run(
    initial_conditions=particles,
    run_time=T_run,
    step_size=1e-2 * cf.Tc.code,
    save_intervals=1000,
    log_intervals=1000,
)
cf.sim.save_data()
zarr.save(data_dir / f"{cf.sim.name}/raw_data/Iw", cf.sim.Iw)
