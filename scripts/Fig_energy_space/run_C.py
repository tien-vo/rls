import config as cf
import zarr
from rls.io import data_dir

cf.sim.name = "fig_energy_space_Bh_03_Bw_001_C"
cf.model.w_wce = 0.15
cf.sim.run(
    initial_conditions=cf.generate_ICs(),
    run_time=cf.T_run,
    step_size=1e-2 * cf.Tc.code,
    save_intervals=3,
    log_intervals=1000,
)
cf.sim.save_data()
zarr.save(data_dir / f"{cf.sim.name}/raw_data/Iw", cf.sim.Iw)
