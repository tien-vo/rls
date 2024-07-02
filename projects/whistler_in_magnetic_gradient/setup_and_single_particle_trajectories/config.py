from astropy.units import Quantity

from rls.data_types import SimQuantity
from rls.formula.conversions import energy_to_speed
from rls.models import WhistlerAtHyperbolicGradientModel
from rls.simulation import Simulation


def get_simulation_energy(energy: Quantity):
    return SimQuantity(model.units.W_factor.user_to_code(energy), energy)


def get_simulation_speed(energy: SimQuantity):
    return energy_to_speed(get_simulation_energy(energy), model.units)


model = WhistlerAtHyperbolicGradientModel(
    Bw=0.05,
    w_wce=0.1,
    sw=1.75,
    Bh=-0.8,
    B0=-1.0,
    theta=0.0,
)
units = model.units
sim = Simulation(model=model)

# ---- Reduce namespace
electron = units.electron
c = units.c
qe = electron.charge
me = electron.mass
eps0 = units.eps0
R = model.R
B0 = model.B0
w_wce = model.w_wce
theta = model.theta
wpe0 = electron.wp(model.n0, units.eps0)
wce0 = electron.wc(model.B0)
Tce0 = electron.Tc(model.B0)
V_factor = (units.L_factor / units.T_factor).to("1000 km/s")
W_factor = units.W_factor
