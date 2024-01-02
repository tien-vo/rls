from time import perf_counter as timer

import numpy as np
import zarr

from rls.data_types import Model, Particle
from rls.integrators.exact_boris import push_r, push_u
from rls.io import data_store


class Simulation:
    def __init__(self, model: Model, name: str | None = None):
        self.name = name
        self.model = model
        self.attrs = {}

    def run(
        self,
        initial_conditions: Particle,
        run_time: float = 1.0,
        step_size: float = 1e-2,
        save_intervals: int = 50,
        log_intervals: int = 20,
    ):
        if not initial_conditions.is_initial_condition:
            raise TypeError("Check initial conditions")
        initial_conditions.check_relativity(self.model.units)

        # Calculate control parameters
        run_steps = np.int64(np.ceil(run_time / step_size))
        save_frequency = max(1, np.int64(np.ceil(run_steps / save_intervals)))
        log_frequency = max(1, np.int64(np.ceil(run_steps / log_intervals)))
        save_steps = np.int64(np.ceil(run_steps / save_frequency))

        # Necessary constants and functions
        species = initial_conditions.species
        push_args = (
            species.charge.code,
            species.mass.code,
            self.model.units.light_speed.code,
            step_size,
        )
        field = self.model.field
        field_args = self.model.field_args
        should_save = lambda n: n % save_frequency == 0
        should_log = lambda n: n % log_frequency == 0

        # Copy particle data to on-memory (and contiguous) arrays
        t = initial_conditions.t
        g = initial_conditions.g.copy()
        x = initial_conditions.x.copy()
        y = initial_conditions.y.copy()
        z = initial_conditions.z.copy()
        ux = initial_conditions.ux.copy()
        uy = initial_conditions.uy.copy()
        uz = initial_conditions.uz.copy()

        # Buffers for on-memory data
        number_of_particles = len(g)
        b_t = np.zeros((save_steps,))
        b_g = np.zeros((save_steps, number_of_particles))
        b_x = np.zeros((save_steps, number_of_particles))
        b_y = np.zeros((save_steps, number_of_particles))
        b_z = np.zeros((save_steps, number_of_particles))
        b_ux = np.zeros((save_steps, number_of_particles))
        b_uy = np.zeros((save_steps, number_of_particles))
        b_uz = np.zeros((save_steps, number_of_particles))

        # Run main loop
        time = timer()
        for n in range(run_steps):
            if should_save(n):
                n_save = n // save_frequency
                b_t[n_save] = t
                b_g[n_save, :] = g
                b_x[n_save, :] = x
                b_y[n_save, :] = y
                b_z[n_save, :] = z
                b_ux[n_save, :] = ux
                b_uy[n_save, :] = uy
                b_uz[n_save, :] = uz

            # ---- First kick: Advance position half a step
            t += step_size / 2
            x = push_r(x, g, ux, step_size / 2)
            y = push_r(y, g, uy, step_size / 2)
            z = push_r(z, g, uz, step_size / 2)

            # ---- Drift: Rotate velocity with the fields at half-step
            Ex, Ey, Ez, Bx, By, Bz = field(t, x, y, z, *field_args)
            g, ux, uy, uz = push_u(
                ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, *push_args
            )

            # ---- Second kick: Advance position the remaining half-step
            t += step_size / 2
            x = push_r(x, g, ux, step_size / 2)
            y = push_r(y, g, uy, step_size / 2)
            z = push_r(z, g, uz, step_size / 2)

            if should_log(n):
                T_per_step = (timer() - time) / log_frequency
                T_est = T_per_step * (run_steps - n) * 0.01668
                print(
                    f"Pushed {n}/{run_steps} steps "
                    f"({n / run_steps:.2%}, "
                    f"{T_per_step:f} ms/step, "
                    f"estimated remaining run time = {T_est:.2f} min)",
                )
                time = timer()

        self.solutions = Particle(
            species, b_t, b_g, b_x, b_y, b_z, b_ux, b_uy, b_uz
        )
        self.attrs = dict(
            run_time=run_time,
            step_size=step_size,
            run_steps=run_steps,
            save_frequency=save_frequency,
            save_steps=save_steps,
            number_of_particles=number_of_particles,
        )
        print("Done!")

    def save_data(self):
        if self.name is None:
            print("`name` attribute not set. Can't save data!")
            return

        kw = dict(store=data_store, path=f"{self.name}/raw_data")
        with zarr.open(mode="a", **kw) as file:
            file.attrs.update(self.attrs)

        zarr.save(t=self.solutions.t, **self.solutions.as_dict(), **kw)

    def load_data(self, species=None):
        if self.name is None:
            print("`name` attribute not set. Can't load data!")
            return

        kw = dict(mode="r", store=data_store)
        def read(variable):
            return zarr.open(path=f"{self.name}/raw_data/{variable}", **kw)

        self.attrs.update(zarr.open(path=f"{self.name}/raw_data", **kw).attrs)
        self.solutions = Particle(
            species,
            read("t")[:],
            read("g")[:],
            read("x")[:],
            read("y")[:],
            read("z")[:],
            read("ux")[:],
            read("uy")[:],
            read("uz")[:],
        )
