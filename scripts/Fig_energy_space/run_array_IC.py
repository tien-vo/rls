import colorcet
import matplotlib as mpl
import numpy as np
from astropy.units import Quantity
from rls.data_types import Particle, SimQuantity
from rls.formula.conversions import cartesian_to_FAC, energy_to_speed
from rls.formula.physics import lorentz_factor
from rls.models import WhistlerInMagneticHoleModel
from rls.simulation import SimulationTrack
from tvolib import mpl_utils as mu


def calculate_V(W, W_factor, units):
    _W = SimQuantity(W_factor.user_to_code(W), W)
    return energy_to_speed(_W, units)


sim = SimulationTrack(
    name="test_wimh_simulation",
    model=WhistlerInMagneticHoleModel(
        Bw=0.05,
        w_wce=0.1,
        zw=-3.5,
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
B0 = model.B0
qe = units.electron.q
me = units.electron.m
wpe0 = units.electron.wp(model.n0, units.eps0)
wce0 = units.electron.wc(model.B0)
Tc = species.cyclotron_period(model.B0)
V_factor = (units.L_factor / units.T_factor).to("1000 km/s")

V_para_min = calculate_V(Quantity(1.0, "eV"), W_factor, units).code
V_para_max = calculate_V(Quantity(400.0, "eV"), W_factor, units).code
V_perp = calculate_V(Quantity(1.0, "eV"), W_factor, units).code
uz0 = np.linspace(V_para_min, V_para_max, 25)
ux0 = np.zeros_like(uz0) + V_perp
uy0 = np.zeros_like(uz0)
x0 = np.zeros_like(uz0)
y0 = np.zeros_like(uz0)
z0 = np.zeros_like(uz0) - 3 * R.code
g0 = lorentz_factor(ux0, uy0, uz0, c.code)
ICs = Particle(species, 0.0, g0, x0, y0, z0, ux0, uy0, uz0)

W_run = Quantity(1.0, "eV")
W_run = SimQuantity(W_factor.user_to_code(W_run), W_run)
V_run = energy_to_speed(W_run, units)
T_run = (6 * R / V_run).code
sim.run(
    initial_conditions=ICs,
    run_time=T_run,
    step_size=1e-2 * Tc.code,
    save_intervals=-1,
    log_intervals=10,
)
# sim.save_data()

t = sim.solutions.t[:]
g, x, y, z, ux, uy, uz = sim.solutions.as_tuple()
tg = t[:, np.newaxis] * np.ones_like(x)

args = model.background_field_args
_, _, _, B0_x, _, B0_z = model.background_field(tg, x, y, z, *args)
V_para, V_perp, W, A = cartesian_to_FAC(g, ux, uy, uz, B0_x, B0_z, model)
W0 = W[0, :]
dW = W - W0[np.newaxis, :]
particle_colors = mpl.colormaps.get_cmap("cet_rainbow4")(
    np.linspace(0, 1, Np := W0.size)
)

V_perp_arr = np.linspace(0, 0.03, 1000) * c.code
z_arr = np.linspace(-3, 3, 1000) * R.code
B0_z_arr = B0.code * model.eta(z_arr, *model.background_field_args)
wce_arr = np.abs(qe.code / me.code * B0_z_arr)
Vr_arr = model.resonant_velocity(wce_arr, model.w_wce, wpe0.code, c.code)
trap_angle = np.degrees(np.arcsin(np.sqrt(B0_z_arr / B0.code)))


fig, axes = mu.plt.subplots(5, 1, figsize=(12, 10), sharex=True)

# 2D domain
_z = np.linspace(-3, 3, Nz := 1000) * R.code
_x = np.linspace(-3, 3, Nx := 1000) * 0.4 * R.code
X, Z = np.meshgrid(_x, _z, indexing="ij")
Y = np.zeros_like(X)
_, _, _, B0_x, _, B0_z = model.background_field(
    0.0, X, Y, Z, *model.background_field_args
)
B0_mag = np.sqrt(B0_x**2 + B0_z**2)
Ew_x, Ew_y, Ew_z, Bw_x, Bw_y, Bw_z = model.wave_field(
    0.0,
    X,
    Y,
    Z,
    B0_x,
    B0_z,
    *model.wave_field_args,
)
cax = mu.add_colorbar(ax := axes[0], size="1%")
ax.streamplot(
    Z / R.code,
    X / R.code,
    B0_z,
    B0_x,
    linewidth=0.2 * B0_mag / B0_mag.min(),
    color="k",
    density=0.5,
)
im = ax.pcolormesh(
    Z / R.code,
    X / R.code,
    Bw_x / B0.code,
    cmap="seismic",
    vmin=-0.06,
    vmax=0.06,
)
cb = fig.colorbar(im, cax=cax, ticks=[-0.05, 0, 0.05])
cb.set_label("$B_w/B_0$")
ax.set_xlim(_z[0] / R.code, _z[-1] / R.code)
ax.set_ylim(_x[0] / R.code, _x[-1] / R.code)
ax.locator_params(axis="y", nbins=5)
ax.set_ylabel("$x/R$")

# 1D domain
mu.add_colorbar(ax := axes[1], size="1%").remove()
ax.plot(Z[Nx // 2, :] / R.code, B0_z[Nx // 2, :] / np.abs(B0.code), "-k")
ax.set_ylabel("$B_{0z}/B_0$")
ax.set_xlim(_z[0] / R.code, _z[-1] / R.code)
ax.locator_params(axis="y", nbins=5)
ax.set_ylim(-1.0, 0)

# z vs Vz
mu.add_colorbar(ax := axes[2], size="1%").remove()
for ip in range(Np):
    ax.plot(
        z[:, ip] / R.code,
        V_para[:, ip].user.value,
        c=particle_colors[ip],
    )

ax.plot(
    z_arr / R.code,
    Vr_arr * V_factor.user.value,
    "--k",
    lw=2,
    zorder=9,
)
ax.set_xlim(_z[0] / R.code, _z[-1] / R.code)
ax.locator_params(axis="y", nbins=5)
ax.set_ylabel(f"$V_\\|$ ({V_factor.user.unit:latex_inline})")

# Pitch angle
mu.add_colorbar(ax := axes[3], size="1%").remove()
ax.fill_between(
    z_arr / R.code,
    trap_angle,
    color="k",
    alpha=0.2,
)
for ip in range(Np):
    ax.plot(
        z[:, ip] / R.code,
        np.degrees(A[:, ip].user.value),
        c=particle_colors[ip],
    )

ax.set_xlim(_z[0] / R.code, _z[-1] / R.code)
ax.locator_params(axis="y", nbins=5)
ax.set_ylabel("$\\alpha$ (deg)")

# Energization
mu.add_colorbar(ax := axes[4], size="1%").remove()
for ip in range(Np):
    ax.plot(
        z[:, ip] / R.code,
        dW[:, ip].user.value,
        c=particle_colors[ip],
    )

ax.set_xlim(_z[0] / R.code, _z[-1] / R.code)
ax.locator_params(axis="y", nbins=5)
ax.set_xlabel("$z/R$")
ax.set_ylabel(f"$\\Delta W$ ({W_factor.user.unit:latex_inline})")

fig.tight_layout(h_pad=0.05)
mu.plt.show()
