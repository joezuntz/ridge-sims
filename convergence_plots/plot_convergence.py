import sys
sys.path.append("..")
import dredge_scms
import ridge_analysis_tools
from mpi4py.MPI import COMM_WORLD
import numpy as np


def generate_ridges_for_convergence():
    base_sim_dir = "../lhc_run_sims"
    coords = ridge_analysis_tools.load_coordinates(base_sim_dir, 1, shift=True, z_cut=0.4)

    n_neighbors = 5000

    # We use the same checkpoint dir to resume from previous runs,
    # so that each run continues from the last one with a tighter convergence
    for convergence in [1e-3, 1e-4, 1e-5]:
        ridges, _, _ = dredge_scms.find_filaments(
            coords,
            bandwidth=np.radians(0.2),
            convergence=np.radians(convergence),
            distance_metric='haversine',
            n_neighbors=n_neighbors,
            comm=COMM_WORLD,
            checkpoint_dir=f"./checkpoints/n_neighbors_{n_neighbors}",
            resume=True,
            seed=12345,
            mesh_size=200000
        )
        if COMM_WORLD.rank == 0:
            np.save(f"./filaments_n_neighbors_{n_neighbors}_convergence_{convergence}.npy", ridges)


def generate_ridges_for_n_neighbors():
    base_sim_dir = "../lhc_run_sims"
    coords = ridge_analysis_tools.load_coordinates(base_sim_dir, 1, shift=True, z_cut=0.4)

    convergence = 1e-4
    n_neighbors_list = [1000, 2000, 3000, 4000]

    # We use the same checkpoint dir to resume from previous runs,
    # so that each run continues from the last one with a tighter convergence
    for n_neighbors in n_neighbors_list:
        ridges, _, _ = dredge_scms.find_filaments(
            coords,
            bandwidth=np.radians(0.2),
            convergence=np.radians(convergence),
            distance_metric='haversine',
            n_neighbors=n_neighbors,
            comm=COMM_WORLD,
            checkpoint_dir=f"./checkpoints/n_neighbors_{n_neighbors}",
            resume=True,
            seed=12345,
            mesh_size=200000
        )
        if COMM_WORLD.rank == 0:
            np.save(f"./filaments_n_neighbors_{n_neighbors}_convergence_{convergence}.npy", ridges)


if __name__ == "__main__":
    # generate_ridges_for_convergence()
    generate_ridges_for_n_neighbors()
