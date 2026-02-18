import os, sys
import numpy as np
import h5py
from mpi4py import MPI

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
os.chdir(current_dir)

import dredge_scms
from ridge_analysis_tools import *

COMM = MPI.COMM_WORLD
RANK = COMM.rank

# this function has a look alike in ridge analysis tools
def run_filament_pipeline_local(bandwidth, base_sim_dir, run_id, base_label, home_dir, N=2, z_cut=0.4, fraction=None, seed=3482364, tag="00"):
    neighbours = 5000
    convergence = 1e-5
    mesh_size = int(N * 5e5)

    os.makedirs(home_dir, exist_ok=True)
    checkpoint_dir = os.path.join(home_dir, f"checkpoints_{tag}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    coordinates = None
    if COMM.rank == 0:
        coordinates = load_coordinates(base_sim_dir, run_id, z_cut=z_cut, fraction=fraction)
    coordinates = COMM.bcast(coordinates, root=0)

    ridges, initial_density, final_density = dredge_scms.find_filaments(
        coordinates,
        bandwidth=np.radians(bandwidth),
        convergence=np.radians(convergence),
        distance_metric='haversine',
        n_neighbors=neighbours,
        comm=COMM,
        checkpoint_dir=checkpoint_dir,
        resume=True,
        seed=seed,
        mesh_size=mesh_size
    )

    COMM.barrier()

    if COMM.rank == 0:
        density_map = build_density_map(base_sim_dir, run_id, 512, z_cut=z_cut)

        plot_dir = os.path.join(home_dir, "plots_by_final_percentile")
        os.makedirs(plot_dir, exist_ok=True)

        fp = 15
        ridges_cut = redo_cuts(
            ridges, initial_density, final_density,
            initial_percentile=0,
            final_percentile=fp
        )

        ridges_dir = os.path.join(home_dir, "Ridges_final_p15")
        os.makedirs(ridges_dir, exist_ok=True)

        ridge_file = os.path.join(ridges_dir, f"{base_label}_run_{run_id}_ridges_p15_{tag}.h5")
        with h5py.File(ridge_file, "w") as f:
            f.create_dataset("ridges", data=ridges_cut)
            f.create_dataset("initial_density", data=initial_density)
            f.create_dataset("final_density", data=final_density)

        plot_path = os.path.join(plot_dir, f"{base_label}_run_{run_id}_Ridges_plot_p15_{tag}.png")
        results_plot(density_map, ridges_cut, plot_path)

    COMM.barrier()


def main():
    N = 2
    bandwidth = 0.1
    run_id = 1

    base_label = "zero_err"
    base_sim_dir = os.path.join(parent_dir, "lhc_run_sims_zero_err_10")

    home_dir = "ridge_variation"
    n_repeats = 50
    base_seed = 123456

    radius_arcmin = 4.0
    min_coverage  = 0.9
    nside         = 512

    mask_filename = os.path.join(parent_dir, "des-data", "desy3_gold_mask.npy")
    mask = np.load(mask_filename) if RANK == 0 else None
    mask = COMM.bcast(mask, root=0)

    ridges_dir = os.path.join(home_dir, "Ridges_final_p15")
    plots_dir  = os.path.join(home_dir, "plots_by_final_percentile")
    if RANK == 0:
        os.makedirs(ridges_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

    for rep in range(n_repeats):
        seed = base_seed + rep
        tag = f"{rep:02d}"

        run_filament_pipeline_local(
            bandwidth=bandwidth,
            base_sim_dir=base_sim_dir,
            run_id=run_id,
            base_label=base_label,
            home_dir=home_dir,
            N=N,
            seed=seed,
            tag=tag
        )

        if RANK == 0:
            ridge_file = os.path.join(ridges_dir, f"{base_label}_run_{run_id}_ridges_p15_{tag}.h5")
            process_ridge_file_local(
                ridge_file,
                mask,
                nside,
                radius_arcmin,
                min_coverage,
                ridges_dir,
                plots_dir
            )


if __name__ == "__main__":
    main()
