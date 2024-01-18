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
    name="fig_setup_array_IC",
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

V_para_min = calculate_V(Quantity(1.0, "eV"), W_factor, units).code
V_para_max = calculate_V(Quantity(400.0, "eV"), W_factor, units).code
V_perp = calculate_V(Quantity(1.0, "eV"), W_factor, units).code
uz0 = np.linspace(V_para_min, V_para_max, 25)
ux0 = np.zeros_like(uz0) + V_perp
uy0 = np.zeros_like(uz0)
x0 = np.zeros_like(uz0)
y0 = np.zeros_like(uz0)
z0 = np.zeros_like(uz0) - 2 * R.code
g0 = lorentz_factor(ux0, uy0, uz0, c.code)
ICs = Particle(species, 0.0, g0, x0, y0, z0, ux0, uy0, uz0)

W_run = Quantity(1.0, "eV")
W_run = SimQuantity(W_factor.user_to_code(W_run), W_run)
V_run = energy_to_speed(W_run, units)
T_run = (4 * R / V_run).code
sim.run(
    initial_conditions=ICs,
    run_time=T_run,
    step_size=1e-2 * Tc.code,
    save_intervals=-1,
    log_intervals=10,
)
sim.save_data()
