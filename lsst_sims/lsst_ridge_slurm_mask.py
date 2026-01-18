import os, sys
import argparse
from numba import config
#config.DISABLE_JIT = True

parser = argparse.ArgumentParser()
parser.add_argument("--task-id", type=int, required=True)
args = parser.parse_args()
run_id = args.task_id

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
os.chdir(current_dir)

import numpy as np
import h5py
import dredge_scms
from mpi4py import MPI
from ridge_analysis_tools import load_coordinates

COMM_WORLD = MPI.COMM_WORLD


def remove_band_around_zero(points: np.ndarray, width_deg: float = 10.0, axis: int = 0) -> np.ndarray:
    """
    Remove points with |points[:, axis]| < width_deg around 0.
    For angles in radians.
    axis=0 -> dec, axis=1 -> ra (given [dec, ra]).
    """
    if points is None:
        return None
    if not (isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 2):
        raise ValueError("Expected points to be an (N,2) numpy array [dec, ra] in radians.")
    w = np.deg2rad(width_deg)
    keep = np.abs(points[:, axis]) >= w
    return points[keep]


def install_mesh_filter(width_deg: float = 10.0, axis: int = 0) -> None:
    """
    the initially generated mesh is filtered before tree-cut + iterations.

    """
    # find_filaments is defined in dredge_scms.main and it imports mesh_generation
    # into its module :contentReference[oaicite:2]{index=2}
    from dredge_scms import main as dredge_main

    original = dredge_main.mesh_generation

    def wrapped_mesh_generation(coordinates, mesh_size, seed=None):
        mesh = original(coordinates, mesh_size, seed)
        # Apply the same zero-line mask to the mesh
        mesh = remove_band_around_zero(mesh, width_deg=width_deg, axis=axis)
        return mesh

    dredge_main.mesh_generation = wrapped_mesh_generation


def run_filament_pipeline(run_id):
    print(f"[rank {COMM_WORLD.rank}] Starting pipeline for run_id={run_id}")
    sys.stdout.flush()

    base_sim_dir = os.path.join(parent_dir, "lhc_run_lsst_sims", "lsst_10")
    out_dir = os.path.join(current_dir, "Temp")

    if COMM_WORLD.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    COMM_WORLD.barrier()

    # ---- Load coords on rank 0
    coords = None
    if COMM_WORLD.rank == 0:
        coords = load_coordinates(
            base_sim_dir,
            run_id,
            z_cut=0.7,
            fraction=0.05
        )

        # Filter the INPUT dataset (coords) 
        coords = remove_band_around_zero(coords, width_deg=10.0, axis=0)  # axis=0 => dec
        print(f"[rank 0] Coords after |dec|>=10deg cut: {coords.shape}")
        sys.stdout.flush()

        # Install the mesh filter *before* calling find_filaments
        install_mesh_filter(width_deg=10.0, axis=0)

    # broadcast filtered coords
    coords = COMM_WORLD.bcast(coords, root=0)

    
    # inside find_filaments. :contentReference[oaicite:3]{index=3}
    

    ridges, initial_density, final_density = dredge_scms.find_filaments(
        coords,
        bandwidth=np.radians(0.4),
        convergence=np.radians(1e-5),
        distance_metric='haversine',
        n_neighbors=600,
        comm=COMM_WORLD,
        checkpoint_dir=os.path.join(out_dir, f"checkpoints_run{run_id}"),
        resume=False,
        seed=12345,
        mesh_size=int(6*5e5),
        mesh_threshold=3.0,
    )

    COMM_WORLD.barrier()

    if COMM_WORLD.rank == 0:
        out_file = os.path.join(out_dir, f"ridges_run{run_id}_band02.h5")
        with h5py.File(out_file, "w") as f:
            f.create_dataset("ridges", data=ridges)
            f.create_dataset("initial_density", data=initial_density)
            f.create_dataset("final_density", data=final_density)
        print(f"[rank 0] Saved results to {out_file}")
        sys.stdout.flush()

    COMM_WORLD.barrier()
    print(f"[rank {COMM_WORLD.rank}] Pipeline finished")
    sys.stdout.flush()


if __name__ == "__main__":
    run_filament_pipeline(run_id)
