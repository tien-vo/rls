import numpy as np
from rls.models import WhistlerInMagneticHoleModel
from rls.simulation import SimulationTrack

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
species = units.scaling_species
c = units.light_speed
R = model.R
W_factor = units.W_factor
Tc = species.Tc(model.B0)
