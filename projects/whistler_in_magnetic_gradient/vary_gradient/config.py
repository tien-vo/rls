import numpy as np
from astropy.units import Quantity

from rls.models import WhistlerAtHyperbolicGradientModel
from rls.simulation import Simulation


def name(ih):
    return f"vary_Bh_{ih}_Bw_0005"


def path(variable):
    return f"vary_Bh_Bw_0005/{variable}"


def calculate_resonant_energy(z_R, V_perp=Quantity(2_000, "km/s"), s=0):
    V_perp = (V_perp / c.user).decompose().value * c.code

    Wr = np.zeros_like(Bh_B0)
    for ih in range(Bh_B0.size):
        model.Bh = Bh_B0[ih] * B0
        B0_z = B0 * model.eta(z_R * R.code, *model.background_field_args)
        wce = np.abs(electron.wc(B0_z)).code
        Vr = model.resonant_velocity(wce, w_wce, wpe0.code, c.code)
        dV = model.resonant_width(
            V_perp, model.Bw.code, B0_z.code, wce, w_wce, wpe0.code, c.code
        )
        Wr[ih] = W_factor.user.value * (0.5 * me.code * (Vr + s * dV) ** 2)

    return Wr


Bh_B0 = np.logspace(-2, 0, 100)

model = WhistlerAtHyperbolicGradientModel(
    Bw=0.005,
    w_wce=0.1,
    sw=1.75,
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
Bw = model.Bw
B0 = model.B0
w_wce = model.w_wce
theta = model.theta
wpe0 = electron.wp(model.n0, units.eps0)
wce0 = electron.wc(model.B0)
Tce0 = electron.Tc(model.B0)
V_factor = (units.L_factor / units.T_factor).to("1000 km/s")
W_factor = units.W_factor

w, k = model.dispersion_relation(wce0.code, w_wce, wpe0.code, c.code)
R_lim = np.abs(Bw.code / B0.code) * (k * R.code)
