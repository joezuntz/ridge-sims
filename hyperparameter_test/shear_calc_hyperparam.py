import os, sys

# Directory of this script 
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add parent directory to python path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Remain inside this directory for execution
os.chdir(current_dir)


import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import re
import time

from ridge_analysis_tools import *
from sklearn.neighbors import NearestNeighbors

from mpi4py import MPI
COMM_WORLD = MPI.COMM_WORLD

try:
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None


# ===============================================================
#   Find contracted files in parameter_test/
# ===============================================================

#def find_contracted_files_parameter_test(root="parameter_test"):
#    out = []
#    for root_dir, dirs, files in os.walk(root):
#        for f in files:
#            if f.endswith("_contracted.h5"):
#                out.append(os.path.join(root_dir, f))
#    return sorted(out)



def find_contracted_files_parameter_test(root="parameter_test"):
    out = []
    for root_dir, dirs, files in os.walk(root):
        for f in files:
            if ("_contracted" in f) and f.endswith(".h5"):
                out.append(os.path.join(root_dir, f))
    return sorted(out)


# ===============================================================
#   Extract mesh size from path
# ===============================================================

def extract_mesh_from_path(path):
    m = re.search(r"mesh_([0-9.]+)", path)
    return m.group(1) if m else None


# ===============================================================
# ============================ MAIN =============================
# ===============================================================

def main():
    contracted_root = "parameter_test/run_1/band_0.1/mesh_8" # this should be adjusted later
    parameter_root = "parameter_test"
    out_root = os.path.join(parameter_root, "shear_vs_meshsize")
    os.makedirs(out_root, exist_ok=True)

    final_percentiles = [15]

    # Background is fixed
    BG_data = os.path.join(
        parent_dir,
        "lhc_run_sims_zero_err_10",
        "run_1",
        "source_catalog_cutzl04.h5"
    )

    # Check background file exists — otherwise nothing to do
    if not os.path.exists(BG_data):
        if comm is None or comm.rank == 0:
            print(f"[FATAL] Missing background file:\n  {BG_data}")
        return

    contracted_files = find_contracted_files_parameter_test(contracted_root) # adjusted to the contracted root above
    if comm is None or comm.rank == 0:
        print(f"Found {len(contracted_files)} contracted ridge files.\n")

    # Safety logs
    skipped_missing_h5 = []
    skipped_missing_background = []  # included for completeness
    skipped_existing_output = []

    # Loop
    for h5_file in contracted_files:
        try:
            # --- Contracted file check ---
            if not os.path.exists(h5_file):
                skipped_missing_h5.append(h5_file)
                if comm is None or comm.rank == 0:
                    print(f"[missing] {h5_file}")
                continue

            # --- Mesh extraction ---
            mesh_value = extract_mesh_from_path(h5_file)
            if mesh_value is None:
                skipped_missing_h5.append(h5_file)
                if comm is None or comm.rank == 0:
                    print(f"[skip] Could not extract mesh value: {h5_file}")
                continue

            # --- Output files ---
            shear_csv     = os.path.join(out_root, f"shear_mesh_{mesh_value}.csv")
            filaments_h5  = os.path.join(out_root, f"filaments_mesh_{mesh_value}.h5")

            # Skip if shear output exists
            if os.path.exists(shear_csv):
                skipped_existing_output.append(shear_csv)
                if comm is None or comm.rank == 0:
                    print(f"[skip] output exists: {shear_csv}")
                continue

            # --- Run processing ---
            process_ridge_file(
                h5_file=h5_file,
                BG_data=BG_data,
                filament_h5=filaments_h5,
                shear_csv=shear_csv,
                background_type = 'sim',
                shear_flip_csv=None,
                comm=comm
            )

        except Exception as e:
            if comm is None or comm.rank == 0:
                print(f"[ERROR] {h5_file} → {e}")

    # ===============================================================
    # Final summary
    # ===============================================================

    if comm is None or comm.rank == 0:
        print("\n================== FINAL SUMMARY ==================\n")

        if skipped_missing_h5:
            print(f"Missing contracted files ({len(skipped_missing_h5)}):")
            for f in skipped_missing_h5:
                print("  -", f)

        if skipped_missing_background:
            print(f"\nMissing background ({len(skipped_missing_background)}):")
            for f in skipped_missing_background:
                print("  -", f)

        if skipped_existing_output:
            print(f"\nSkipped existing outputs ({len(skipped_existing_output)}):")
            for f in skipped_existing_output:
                print("  -", f)

        if not (skipped_missing_h5 or skipped_missing_background or skipped_existing_output):
            print("No files were skipped.")

        print("\n==================================================")
        print("All shear calculations completed.")


if __name__ == "__main__":
    main()
