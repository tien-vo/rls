import numpy as np
from astropy.units import Quantity

from rls.models import WhistlerAtHyperbolicGradientModel
from rls.simulation import Simulation


def name(ib):
    return f"vary_Bw_{ib}_Bh_03"


def path(variable):
    return f"vary_Bw_Bh_03/{variable}"


def calculate_resonant_energy(z_R, V_perp=Quantity(4_000, "km/s"), s=0):
    V_perp = (V_perp / c.user).decompose().value * c.code

    Wr = np.zeros_like(Bw_B0)
    for ib in range(Bw_B0.size):
        model.Bw = np.abs(Bw_B0[ib] * B0)
        B0_z = B0 * model.eta(z_R * R.code, *model.background_field_args)
        wce = np.abs(electron.wc(B0_z)).code
        Vr = model.resonant_velocity(wce, w_wce, wpe0.code, c.code)
        dV = model.resonant_width(
            V_perp, model.Bw.code, B0_z.code, wce, w_wce, wpe0.code, c.code
        )
        Wr[ib] = W_factor.user.value * (0.5 * me.code * (Vr + s * dV) ** 2)

    return Wr


def calculate_resonant_velocity(V_perp, Bw_B0, Bh_B0, z_R):
    model.Bw = np.abs(Bw_B0 * B0)
    model.Bh = Bh_B0 * B0

    B0_z = B0 * model.eta(
        z_R * R.code,
        *model.background_field_args,
    )
    wce = np.abs(electron.wc(B0_z)).code
    Vr = model.resonant_velocity(wce, w_wce, wpe0.code, c.code)
    dV = model.resonant_width(
        V_perp, model.Bw.code, B0_z.code, wce, w_wce, wpe0.code, c.code
    )
    return Vr, Vr + dV, Vr - dV


Bw_B0 = np.logspace(np.log10(1e-4), np.log10(5e-1), 100)

model = WhistlerAtHyperbolicGradientModel(
    w_wce=0.1,
    sw=1.75,
    Bh=-0.3,
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
Bh = model.Bh
B0 = model.B0
w_wce = model.w_wce
theta = model.theta
wpe0 = electron.wp(model.n0, units.eps0)
wce0 = electron.wc(model.B0)
Tce0 = electron.Tc(model.B0)
V_factor = (units.L_factor / units.T_factor).to("1000 km/s")
W_factor = units.W_factor

w, k = model.dispersion_relation(wce0.code, w_wce, wpe0.code, c.code)
R_lim = (Bh.code / B0.code) / (k * R.code)
