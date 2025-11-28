
# the folder structure will be as follows: This folder should be made inside /lsst_sims

"""
LSST_ridges/
    lhc_run_lsst_sims/
        lsst_1/
            run_1/
                band_0.1/
                    Ridges_final_p15.h5
            run_2/
                band_0.1/
                    ...
        lsst_10/
            ...
    lhc_run_lsst_sim_zero_err/
        lsst_1/
            run_1/
                band_0.1/
                    Ridges_final_p15.h5
                    
                    
                    
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



# ------------------------------------------------------------
# Main execusion 
# ------------------------------------------------------------
def main():

    # Simulation types 
    sim_roots = [
        os.path.join(current_dir, "lhc_run_lsst_sims"),
        os.path.join(current_dir, "lhc_run_lsst_sim_zero_err"),
    ]

    output_base = "LSST_ridges"
    os.makedirs(output_base, exist_ok=True)

    bandwidth = 0.1
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
            expected_output = os.path.join(home_dir, "Ridges_final_p15.h5")
            if os.path.exists(expected_output):
                if COMM_WORLD.rank == 0:
                    print(f"[INFO] Output exists → skipping {lsst_label}/run_{run_id}")
                skipped_runs.append(f"{lsst_label}/run_{run_id}")
                continue

            # ------------------------------------------------------------
            # Filament pipeline
            # ------------------------------------------------------------
            run_filament_pipeline(
                bandwidth=bandwidth,
                base_sim_dir=base_sim_dir,
                run_ids=[run_id],
                base_label=base_label,
                home_dir=home_dir
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
