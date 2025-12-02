import os, sys
import argparse

# Environment setup

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

#stay inside the directory for output
os.chdir(current_dir)

# Project imports
import numpy as np
import h5py
import dredge_scms
import healpy
import matplotlib.pyplot as plt
from ridge_analysis_tools import *
from mpi4py import MPI

COMM_WORLD = MPI.COMM_WORLD

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None


# Filament pipeline

def run_filament_pipeline_local(bandwidth, base_sim_dir, run_ids, base_label, home_dir, N=2, z_cut=0.7):

    neighbours = 5000
    convergence = 1e-5
    seed = 3482364
    mesh_size = int(N * 5e5)

    os.makedirs(home_dir, exist_ok=True)
    checkpoint_dir = os.path.join(home_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for run_id in run_ids:

        # Load only on rank 0
        coordinates = None
        if COMM_WORLD.rank == 0:
            coordinates = load_coordinates(base_sim_dir, run_id, z_cut=z_cut)

        # Broadcast to all ranks
        coordinates = COMM_WORLD.bcast(coordinates, root=0)

        # Run SCMS
        ridges, initial_density, final_density = dredge_scms.find_filaments(
            coordinates,
            bandwidth=np.radians(bandwidth),
            convergence=np.radians(convergence),
            distance_metric='haversine',
            n_neighbors=neighbours,
            comm=COMM_WORLD,
            checkpoint_dir=checkpoint_dir,
            resume=True,
            seed=seed,
            mesh_size=mesh_size
        )

        COMM_WORLD.barrier()

        # Rank 0 writes output
        if COMM_WORLD.rank == 0:

            final_percentiles = [15]
            initial_percentile = 0

            density_map = build_density_map(base_sim_dir, run_id, 512, z_cut=z_cut)

            plot_dir = os.path.join(home_dir, "plots_by_final_percentile")
            os.makedirs(plot_dir, exist_ok=True)

            for fp in final_percentiles:

                ridges_cut = redo_cuts(
                    ridges, initial_density, final_density,
                    initial_percentile=initial_percentile,
                    final_percentile=fp
                )

                out_dir = os.path.join(home_dir, f"Ridges_final_p{fp:02d}")
                os.makedirs(out_dir, exist_ok=True)

                h5_filename = os.path.join(
                    out_dir,
                    f"{base_label}_run_{run_id}_ridges_p{fp:02d}_z{z_cut:.2f}.h5"
                )

                with h5py.File(h5_filename, 'w') as f:
                    f.create_dataset("ridges", data=ridges_cut)
                    f.create_dataset("initial_density", data=initial_density)
                    f.create_dataset("final_density", data=final_density)

                print(f"[rank 0] Saved ridges → {h5_filename}")

                plot_path = os.path.join(
                    plot_dir,
                    f"{base_label}_run_{run_id}_Ridges_plot_p{fp:02d}_z{z_cut:.2f}.png"
                )
                results_plot(density_map, ridges_cut, plot_path)
                print(f"[rank 0] Saved plot: {plot_path}")

        COMM_WORLD.barrier()



# Main 
def main():

    # ------------------------------------
    # Slurm array index support
    # ------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=None)
    args = parser.parse_args()
    task_id = args.task_id  # Here we have 1–8 from the Slurm array

    # Two simulation types
    sim_roots = [
        os.path.join(parent_dir, "lsst_sim_ridges"),
        os.path.join(parent_dir, "lsst_sim_zero_err_ridges"),
    ]

    output_base = "LSST_ridges"
    os.makedirs(output_base, exist_ok=True)

    bandwidth = 0.2
    skipped_runs = []
    missing_inputs = []

    # ------------------------------------
    # Discover ALL (sim_root, lsst_label, run_id) as a flat list
    # ------------------------------------
    all_jobs = []

    for sim_root in sim_roots:

        discovered = discover_lsst_runs(sim_root)
        discovered = sorted(discovered, key=lambda x: (x[0], x[1]))

        sim_root_label = os.path.basename(sim_root.rstrip("/"))

        for (lsst_label, run_id, run_path) in discovered:
            all_jobs.append((sim_root, sim_root_label, lsst_label, run_id))

    # ------------------------------------
    # If Slurm task-id is given → select exactly one job
    # ------------------------------------
    if task_id is not None:
        if task_id < 1 or task_id > len(all_jobs):
            raise ValueError(f"Invalid --task-id {task_id}, only 1–{len(all_jobs)} exist.")

        all_jobs = [all_jobs[task_id - 1]]  # Select single job

    # ------------------------------------
    # Run selected jobs (usually exactly one)
    # ------------------------------------
    for sim_root, sim_root_label, lsst_label, run_id in all_jobs:

        if COMM_WORLD.rank == 0:
            print(f"\n[rank 0] Processing → {sim_root_label}/{lsst_label}/run_{run_id}")

        base_label = lsst_label
        base_sim_dir = os.path.join(sim_root, lsst_label)

        home_dir = os.path.join(
            output_base,
            sim_root_label,
            lsst_label,
            f"run_{run_id}",
            f"band_{bandwidth:.1f}"
        )
        os.makedirs(home_dir, exist_ok=True)

        # Check input existence
        input_file = os.path.join(base_sim_dir, f"run_{run_id}", "lens_catalog_0.npy")
        if not os.path.exists(input_file):
            if COMM_WORLD.rank == 0:
                print(f"[WARN] Missing input: {input_file} → skipping.")
            missing_inputs.append(f"{lsst_label}/run_{run_id}")
            continue

        # Check output existence (corrected f-string)
        expected_output = os.path.join(
            home_dir,
            f"Ridges_final_p15_z{0.70:.2f}.h5"
        )
        if os.path.exists(expected_output):
            if COMM_WORLD.rank == 0:
                print(f"[INFO] Output already exists → skipping {lsst_label}/run_{run_id}")
            skipped_runs.append(f"{lsst_label}/run_{run_id}")
            continue

        # Run filament pipeline
        run_filament_pipeline_local(
            bandwidth=bandwidth,
            base_sim_dir=base_sim_dir,
            run_ids=[run_id],
            base_label=base_label,
            home_dir=home_dir,
            z_cut=0.7
        )

    # ------------------------------------
    # Final summary of any missing input
    # ------------------------------------
    if COMM_WORLD.rank == 0:
        print("\nAll runs completed.")

        if skipped_runs:
            print("\nSkipped runs (output existed):")
            for r in skipped_runs:
                print("  -", r)

        if missing_inputs:
            print("\nMissing input files:")
            for r in missing_inputs:
                print("  -", r)

        print("\nDone.\n")


if __name__ == "__main__":
    main()
