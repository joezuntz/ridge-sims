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



import pandas as pd
import numpy as np
import h5py
import time
from ridge_analysis_tools import *
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import re

from mpi4py import MPI

COMM_WORLD = MPI.COMM_WORLD

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None





# After execution the folder should look like 

"""
Cosmo_sim_ridges/S8/run_1/band_0.1/shear/
    ├── filaments_p15.h5
    ├── shear_p15.csv
    └── shear_p15_flipG1.csv  # If not None
"""

# Temporary helper
def find_contracted_files_update(home_dir):
    """find all '_contracted.h5' ridge files in directory."""
    contracted_files = []
    for root, _, files in os.walk(home_dir):
        for f in files:
            if f.endswith("_contracted_update.h5"):
                contracted_files.append(os.path.join(root, f))
    return contracted_files


# ===============================================================
# ========================== MAIN ===============================
# ===============================================================

def main():
    # --- Root directories ---
    home_dir = "Cosmo_sim_ridges"
    base_sim_root = os.path.abspath(os.path.join(parent_dir, "lhc_cosmo_sims_zero_err")) 
    final_percentiles = [15]

    # --- Find contracted ridge files ---
    contracted_files = find_contracted_files_update(home_dir)                          # update
    if comm is None or COMM_WORLD.rank == 0:
        print(f"Found {len(contracted_files)} contracted ridge files.\n")

    # ===============================================================
    # SAFETY MECHANISM 
    # ===============================================================
    skipped_missing_input = []          # Background missing
    skipped_missing_h5 = []             # Contracted ridge file missing
    skipped_existing_output = []        # Output already exists

    # --- Loop over ridge files ---
    for h5_file in contracted_files:
        try:
            # --- SAFETY CHECK: contracted ridge file missing ---
            if not os.path.exists(h5_file):
                skipped_missing_h5.append(h5_file)
                if comm is None or COMM_WORLD.rank == 0:
                    print(f"[missing] Ridge file not found: {h5_file}")
                continue

            # --- Find matching background ---
            BG_data = find_background_file(h5_file, base_sim_root)
            if BG_data is None or not os.path.exists(BG_data):
                # SAFETY MECHANISM — skip if background missing
                skipped_missing_input.append(h5_file)
                if comm is None or COMM_WORLD.rank == 0:
                    print(f"[skip] Missing background for {h5_file}")
                continue

            # --- Build output dirs ---
            band_dir = os.path.dirname(h5_file)
            shear_dir = os.path.join(band_dir, "shear_update")                            #Update
            os.makedirs(shear_dir, exist_ok=True)

            # --- Loop over percentiles ---
            for fp in final_percentiles:
                filament_h5 = os.path.join(shear_dir, f"filaments_p{fp:02d}.h5")
                shear_csv = os.path.join(shear_dir, f"shear_p{fp:02d}.csv")
                shear_flip_csv = os.path.join(shear_dir, f"shear_p{fp:02d}_flipG1.csv")

                # SAFETY MECHANISM — skip if output already exists
                if os.path.exists(shear_csv):
                    skipped_existing_output.append(shear_csv)
                    if comm is None or COMM_WORLD.rank == 0:
                        print(f"[skip] {shear_csv} already exists.")
                    continue

                # --- Normal processing ---
                process_ridge_file(
                    h5_file=h5_file,
                    BG_data=BG_data,
                    filament_h5=filament_h5,
                    shear_csv=shear_csv,
                    shear_flip_csv=None,
                    comm=comm
                )

        except Exception as e:
            if comm is None or COMM_WORLD.rank == 0:
                print(f"[ERROR] Skipping {h5_file}: {e}")

    # ===============================================================
    #  -------------- SUMMARY -----------------------------
    # ===============================================================
    if comm is None or COMM_WORLD.rank == 0:
        print("\n================== FINAL SUMMARY ==================\n")

        if skipped_missing_h5:
            print(f"Missing ridge input files ({len(skipped_missing_h5)}):")
            for f in skipped_missing_h5:
                print(f"  - {f}")

        if skipped_missing_input:
            print(f"\nSkipped (missing background input) ({len(skipped_missing_input)}):")
            for f in skipped_missing_input:
                print(f"  - {f}")

        if skipped_existing_output:
            print(f"\nSkipped (output already existed) ({len(skipped_existing_output)}):")
            for f in skipped_existing_output:
                print(f"  - {f}")

        if not (skipped_missing_h5 or skipped_missing_input or skipped_existing_output):
            print("No files were skipped.")

        print("\n==================================================")
        print("\nAll shear calculations complete.")


if __name__ == "__main__":
    main()