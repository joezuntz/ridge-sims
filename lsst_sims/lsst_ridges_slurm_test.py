import os, sys
import argparse


# Parse SLURM array task ID
parser = argparse.ArgumentParser()
parser.add_argument("--task-id", type=int, required=True)
args = parser.parse_args()

run_id = args.task_id   # run = array index


# Environment setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

os.chdir(current_dir)


# Imports
import numpy as np
import h5py
import healpy
import dredge_scms
from mpi4py import MPI
from ridge_analysis_tools import load_coordinates

COMM_WORLD = MPI.COMM_WORLD


def run_filament_pipeline(run_id):

    # ----------------------------
    # Pipeline start
    # ----------------------------
    print(f"[rank {COMM_WORLD.rank}] Starting pipeline for run_id={run_id}")
    sys.stdout.flush()

    # Simulation directory
    base_sim_dir = os.path.join(parent_dir, "lhc_run_lsst_sims", "lsst_10")

    # Output directory
    out_dir = os.path.join(current_dir, "Temp")
    if COMM_WORLD.rank == 0:
        print(f"[rank 0] Creating output directory: {out_dir}")
        sys.stdout.flush()
        os.makedirs(out_dir, exist_ok=True)

    COMM_WORLD.barrier()
    print(f"[rank {COMM_WORLD.rank}] Passed output-dir barrier")
    sys.stdout.flush()

    # ----------------------------
    # Load coordinates
    # ----------------------------
    coords = None
    if COMM_WORLD.rank == 0:
        print(f"[rank 0] Loading coords from: {base_sim_dir}/run_{run_id}")
        sys.stdout.flush()

        coords = load_coordinates(
            base_sim_dir,
            run_id,
            z_cut=0.7,
            fraction=0.25
        )

        print(f"[rank 0] Loaded coords with shape: {coords.shape}")
        sys.stdout.flush()

    coords = COMM_WORLD.bcast(coords, root=0)

    print(f"[rank {COMM_WORLD.rank}] Received coords")
    sys.stdout.flush()

    # ----------------------------
    # Finding ridges
    # ----------------------------
    print(f"[rank {COMM_WORLD.rank}] Starting filament finding")
    sys.stdout.flush()

    ridges, initial_density, final_density = dredge_scms.find_filaments(
        coords,
        bandwidth=np.radians(0.2),
        convergence=np.radians(1e-5),
        distance_metric='haversine',
        n_neighbors=3000,
        comm=COMM_WORLD,
        checkpoint_dir=os.path.join(out_dir, f"checkpoints_run{run_id}"),
        resume=False,
        seed=12345,
        mesh_size=200000
    )

    print(f"[rank {COMM_WORLD.rank}] Finished filament finding")
    sys.stdout.flush()

    COMM_WORLD.barrier()
    print(f"[rank {COMM_WORLD.rank}] Passed post-filament barrier")
    sys.stdout.flush()

    # ----------------------------
    # Save results
    # ----------------------------
    if COMM_WORLD.rank == 0:
        out_file = os.path.join(out_dir, f"ridges_run{run_id}_band02.h5")
        print(f"[rank 0] Saving results to {out_file}")
        sys.stdout.flush()

        with h5py.File(out_file, "w") as f:
            f.create_dataset("ridges", data=ridges)
            f.create_dataset("initial_density", data=initial_density)
            f.create_dataset("final_density", data=final_density)

        print(f"[rank 0] Saved results successfully")
        sys.stdout.flush()

    COMM_WORLD.barrier()
    print(f"[rank {COMM_WORLD.rank}] Pipeline finished")
    sys.stdout.flush()


if __name__ == "__main__":
    run_filament_pipeline(run_id)
