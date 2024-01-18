from rls.models import WhistlerAtHyperbolicGradientModel
from rls.simulation import Simulation

sim = Simulation(
    model=WhistlerAtHyperbolicGradientModel(
        Bw=0.05,
        w_wce=0.1,
        sw=1.75,
        Bh=0.8,
        B0=1.0,
        theta=0.0,
    )
)
model = sim.model
units = model.units
species = units.electron
W_factor = units.W_factor
V_factor = (units.L_factor / units.T_factor).to("1000 km/s")

c = units.c
R = model.R
B0 = model.B0
w_wce = model.w_wce
theta = model.theta
xw = model.xw
zw = model.zw
sw = model.sw
qe = species.q
me = species.m
wpe0 = species.wp(model.n0, units.eps0)
Tc = species.cyclotron_period(model.B0)
