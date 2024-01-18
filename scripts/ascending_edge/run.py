import config as cf
import numpy as np
from astropy.units import Quantity
from rls.data_types import Particle, SimQuantity
from rls.formula.conversions import energy_to_speed
from rls.formula.physics import lorentz_factor


def calculate_V(W):
    _W = SimQuantity(cf.W_factor.user_to_code(W), W)
    return energy_to_speed(_W, cf.units)


V_para_min = -calculate_V(Quantity(400.0, "eV")).code
V_para_max = -calculate_V(Quantity(1.0, "eV")).code
V_perp = calculate_V(Quantity(1.0, "eV")).code
uz0 = np.linspace(V_para_min, V_para_max, 25)
ux0 = np.zeros_like(uz0) + V_perp
uy0 = np.zeros_like(uz0)
x0 = np.zeros_like(uz0)
y0 = np.zeros_like(uz0)
z0 = np.zeros_like(uz0) + 2 * cf.R.code
g0 = lorentz_factor(ux0, uy0, uz0, cf.c.code)
ICs = Particle(cf.species, 0.0, g0, x0, y0, z0, ux0, uy0, uz0)

W_run = Quantity(1.0, "eV")
W_run = SimQuantity(cf.W_factor.user_to_code(W_run), W_run)
V_run = energy_to_speed(W_run, cf.units)
T_run = (8 * cf.R / V_run).code
cf.sim.name = "ascending_edge"
cf.sim.run(
    initial_conditions=ICs,
    run_time=T_run,
    step_size=1e-2 * cf.Tc.code,
    save_intervals=-1,
    log_intervals=10,
)
cf.sim.save_data()
