import numpy as np
from rls.models import WhistlerAtHyperbolicGradientModel
from rls.simulation import Simulation

Bw_B0 = np.logspace(np.log10(1e-4), np.log10(5e-1), 100)
name = lambda ib: f"vary_Bw_{ib}_Bh_03"
process_path = "Fig_vary_Bw_Bh_03"

sim = Simulation(
    model=WhistlerAtHyperbolicGradientModel(
        w_wce=0.1,
        sw=1.75,
        Bh=-0.3,
        B0=-1.0,
        theta=0.0,
    )
)
model = sim.model
units = model.units
species = units.scaling_species
c = units.light_speed
R = model.R
Bh = model.Bh
B0 = model.B0
w_wce = model.w_wce
theta = model.theta
wpe0 = units.electron.wp(model.n0, units.eps0)
wce0 = units.electron.wc(model.B0)
Tc = species.Tc(model.B0)
V_factor = (units.L_factor / units.T_factor).to("1000 km/s")
W_factor = units.W_factor

w, k = model.dispersion_relation(wce0.code, w_wce, wpe0.code, c.code)
R_lim = 1 / k / R.code * (Bh.code / B0.code)
