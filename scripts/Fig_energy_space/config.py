import numpy as np
from astropy.units import Quantity
from rls.data_types import Particle, SimQuantity
from rls.formula.conversions import energy_to_speed
from rls.formula.physics import lorentz_factor
from rls.models import WhistlerInMagneticHoleModel
from rls.simulation import SimulationTrack


def calculate_V(W, W_factor, units):
    _W = SimQuantity(W_factor.user_to_code(W), W)
    return energy_to_speed(_W, units)


def generate_ICs():
    rng = np.random.default_rng()
    number_of_particles = 1_000_000
    W = 10.0 ** rng.uniform(np.log10(50), np.log10(2050), number_of_particles)
    A = rng.uniform(0.0, np.radians(180), number_of_particles)
    phase = rng.uniform(-np.pi, np.pi, number_of_particles)
    V = calculate_V(Quantity(W, "eV"), W_factor, units).code
    uz0 = V * np.cos(A)
    ux0 = V * np.sin(A) * np.cos(phase)
    uy0 = V * np.sin(A) * np.sin(phase)
    x0 = np.zeros_like(uz0)
    y0 = np.zeros_like(uz0)
    z0 = np.zeros_like(uz0) - 2 * R.code
    g0 = lorentz_factor(ux0, uy0, uz0, c.code)
    return Particle(species, 0.0, g0, x0, y0, z0, ux0, uy0, uz0)


sim = SimulationTrack(
    model=WhistlerInMagneticHoleModel(
        Bw=0.01,
        zw=-3.5,
        sw=1.75,
        Bh=-0.5,
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
V_run = calculate_V(Quantity(1.0, "eV"), W_factor, units)
T_run = (4 * R / V_run).code
