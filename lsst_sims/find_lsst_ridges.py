
# the folder structure will be as follows: This folder should be made inside /lsst_sims

"""
LSST_ridges/
    lhc_run_lsst_sims/
        lsst_1/
            run_1/
                band_X/
                    Ridges_final_p15_z_cut.h5
            run_2/
                band_X/
                    ...
        lsst_10/
            ...
    lhc_run_lsst_sim_zero_err/

                ...                    
                    
"""

import os, sys

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# stay in current directories
os.chdir(current_dir)


#Project imports
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


# Temporary local pipeline to test for different values of the redshift 

def run_filament_pipeline_local(bandwidth, base_sim_dir, run_ids, base_label, home_dir, N = 2, z_cut=0.4):
    """
    Run the full filament-finding + plotting for a given bandwidth, simulation base, and run IDs.
    Results are grouped under the same bandwidth + base label directory.
    z_cut=0.4 is default for DES sims. It is to be changed for lsst sims
    """
    # --- Parameters ---
    neighbours = 5000
    convergence = 1e-5
    seed = 3482364
    mesh_size = int(N * 5e5)

    # --- Directory structure ---
    os.makedirs(home_dir, exist_ok=True)
    checkpoint_dir = os.path.join(home_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for run_id in run_ids:
        # --- Load coordinates on rank 0 only ---
        coordinates = None
        if COMM_WORLD.rank == 0:
            coordinates = load_coordinates(base_sim_dir, run_id, z_cut=z_cut )

        # --- Broadcast to all ranks ---
        coordinates = COMM_WORLD.bcast(coordinates, root=0)

        # --- Parallelized filament finder ---
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

        # --- Synchronize all ranks before output ---
        COMM_WORLD.barrier()

        # --- Output (rank 0 only) ---
        if COMM_WORLD.rank == 0:
            final_percentiles = [15]
            initial_percentile = 0

            # Build the density map (rank 0 only)
            density_map = build_density_map(base_sim_dir, run_id, 512, z_cut=z_cut)

            plot_dir = os.path.join(home_dir, "plots_by_final_percentile")
            os.makedirs(plot_dir, exist_ok=True)

            for fp in final_percentiles:
                ridges_cut = redo_cuts(
                    ridges, initial_density, final_density,
                    initial_percentile=initial_percentile,
                    final_percentile=fp
                )

                # Save ridges (by run ID)
                out_dir = os.path.join(home_dir, f"Ridges_final_p{fp:02d}")
                os.makedirs(out_dir, exist_ok=True)
                h5_filename = os.path.join(out_dir, f"{base_label}_run_{run_id}_ridges_p{fp:02d}_z{z_cut:.2f}.h5")

                with h5py.File(h5_filename, 'w') as f:
                    f.create_dataset("ridges", data=ridges_cut)
                    f.create_dataset("initial_density", data=initial_density)
                    f.create_dataset("final_density", data=final_density)

                print(f"[rank 0] Saved ridges → {h5_filename}")

                # Plot
                plot_path = os.path.join(plot_dir, f"{base_label}_run_{run_id}_Ridges_plot_p{fp:02d}_z{z_cut:.2f}.png")
                results_plot(density_map, ridges_cut, plot_path)
                print(f"[rank 0] Saved plot: {plot_path}")

        # --- Synchronize before next run_id ---
        COMM_WORLD.barrier()














# ------------------------------------------------------------
# Main execusion 
# ------------------------------------------------------------
def main():

    # Simulation types 
    sim_roots = [
    os.path.join(parent_dir, "lhc_run_lsst_sims"),
    os.path.join(parent_dir, "lhc_run_lsst_sim_zero_err"),
    ]

    output_base = "LSST_ridges"
    os.makedirs(output_base, exist_ok=True)

    bandwidth = 0.2
    skipped_runs = []
    missing_inputs = []

    for sim_root in sim_roots:

        sim_root_label = os.path.basename(sim_root.rstrip("/"))

        if COMM_WORLD.rank == 0:
            print(f"[rank 0] Scanning simulation root: {sim_root}")

        discovered = discover_lsst_runs(sim_root)

        if COMM_WORLD.rank == 0:
            print(f"[rank 0] Found {len(discovered)} valid run directories.")

        for lsst_label, run_id, run_path in discovered:

            if COMM_WORLD.rank == 0:
                print(f"\n[rank 0] Starting ridge extraction for {lsst_label}/run_{run_id}")

            base_label = lsst_label
            base_sim_dir = os.path.join(sim_root, lsst_label)

            # ------------------------------------------------------------
            # We should expect this structure for the output: 
            # LSST_ridges/<sim_root_label>/lsst_X/run_Y/band_0.1/
            # ------------------------------------------------------------
            home_dir = os.path.join(
                output_base,
                sim_root_label,
                lsst_label,
                f"run_{run_id}",
                f"band_{bandwidth:.1f}"
            )
            os.makedirs(home_dir, exist_ok=True)

            # Check missing input
            input_file = os.path.join(base_sim_dir, f"run_{run_id}", "lens_catalog_0.npy")
            if not os.path.exists(input_file):
                if COMM_WORLD.rank == 0:
                    print(f"[WARN] Missing input: {input_file}. Skipping run.")
                missing_inputs.append(f"{lsst_label}/run_{run_id}")
                continue

            # Check if output exists
            expected_output = os.path.join(home_dir, "Ridges_final_p15_z{z_cut:.2f}.h5")
            if os.path.exists(expected_output):
                if COMM_WORLD.rank == 0:
                    print(f"[INFO] Output exists → skipping {lsst_label}/run_{run_id}")
                skipped_runs.append(f"{lsst_label}/run_{run_id}")
                continue

            # ------------------------------------------------------------
            # Filament pipeline
            # ------------------------------------------------------------
            run_filament_pipeline_local(
                bandwidth=bandwidth,
                base_sim_dir=base_sim_dir,
                run_ids=[run_id],
                base_label=base_label,
                home_dir=home_dir,
                z_cut=0.7
            )

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------
    if COMM_WORLD.rank == 0:

        print("All runs completed.")


        if skipped_runs:
            print("\nSkipped runs (already had outputs):")
            for r in skipped_runs:
                print(f"   - {r}")

        if missing_inputs:
            print("\nMissing input files:")
            for r in missing_inputs:
                print(f"   - {r}")

        print("\nDone.\n")


if __name__ == "__main__":
    main()
