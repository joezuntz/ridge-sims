import os, sys

# Directory of this script (cosmo_sims)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up 
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# find modules in the parent directory
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# we stay inside cosmo_sims for outputs
os.chdir(current_dir)

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





def main():
    base_sim_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "lhc_cosmo_sims_zero_err"))
    output_base = "Cosmo_sim_ridges"
    os.makedirs(output_base, exist_ok=True)
    categories = [ "S8_perp", "Om_fixed", "sigma8_fixed", "S8"]
    num_runs = 10
    bandwidth = 0.1


    skipped_runs = []
    missing_inputs = []

    for category in categories:
        for run_idx in range(1, num_runs + 1):
            if COMM_WORLD.rank == 0:
                print(f"\n[rank 0] Starting ridge extraction for {category}/run_{run_idx}")

            sim_dir = os.path.join(base_sim_dir, category)
            base_label = category  # for filenames inside the function

            # New directory structure
            home_dir = os.path.join(output_base, category, f"run_{run_idx}", f"band_{bandwidth:.1f}")
            os.makedirs(home_dir, exist_ok=True)
            
            
            # --- Check for missing input file --- 
            # The aim is for the code to continue for now but there will be a summary to flag everything that was ignored
            input_file = os.path.join(sim_dir, f"run_{run_idx}", "lens_catalog_0.npy")
            if not os.path.exists(input_file):
                if COMM_WORLD.rank == 0:
                    print(f"[WARN] Missing input file: {input_file}. Skipping run.")
                missing_inputs.append(f"{category}/run_{run_idx}")
                continue

            # --- Check for existing output file ---
            # not to run the same thing twice
            output_file = os.path.join(home_dir, "Ridges_final_p15.h5")
            if os.path.exists(output_file):
                if COMM_WORLD.rank == 0:
                    print(f"[INFO] Output already exists for {category}/run_{run_idx}. Skipping.")
                skipped_runs.append(f"{category}/run_{run_idx}")
                continue
            
            
            run_filament_pipeline(
                bandwidth=bandwidth,
                base_sim_dir=sim_dir,
                run_ids=[run_idx],
                base_label=base_label,
                home_dir=home_dir
            )

    # === Final summary ===
    if COMM_WORLD.rank == 0:
        print("\nAll ridge extraction runs complete.\n")
        if skipped_runs:
            print("Skipped (already existing outputs):")
            for r in skipped_runs:
                print(f"   - {r}")
        else:
            print("No skipped runs (all new).")

        if missing_inputs:
            print("\nMissing input files (ignored):")
            for r in missing_inputs:
                print(f"   - {r}")
        else:
            print("\nNo missing inputs.")

        print("\nDone.\n")


if __name__ == "__main__":
    main()















