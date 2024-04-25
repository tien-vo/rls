import numpy as np
from rls.models import WhistlerAtHyperbolicGradientModel
from rls.simulation import Simulation

Bh_B0 = np.logspace(-2, 0, 100)
name = lambda ih: f"vary_Bh_{ih}_Bw_0005"
process_path = "Fig_vary_Bh_Bw_0005"

sim = Simulation(
    model=WhistlerAtHyperbolicGradientModel(
        Bw=0.005,
        w_wce=0.1,
        sw=1.75,
        B0=-1.0,
        theta=0.0,
    )
)
model = sim.model
units = model.units
species = units.scaling_species
c = units.light_speed
R = model.R
Bw = model.Bw
B0 = model.B0
w_wce = model.w_wce
theta = model.theta
wpe0 = units.electron.wp(model.n0, units.eps0)
wce0 = units.electron.wc(model.B0)
Tc = species.Tc(model.B0)
V_factor = (units.L_factor / units.T_factor).to("1000 km/s")
W_factor = units.W_factor

w, k = model.dispersion_relation(wce0.code, w_wce, wpe0.code, c.code)
R_lim = k * R.code * np.abs(Bw.code / B0.code)
