import os, sys

# ===============================================================
# PATH SETUP
# ===============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, '..'))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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
comm = MPI.COMM_WORLD
rank = comm.rank


# ===============================================================
#   Find contracted files (mesh fixed, bandwidth scanned)
# ===============================================================
def find_contracted_files_parameter_test(root="parameter_test"):
    out = []
    for root_dir, dirs, files in os.walk(root):
        for f in files:
            if f.endswith("_contracted.h5"):
                out.append(os.path.join(root_dir, f))
    return sorted(out)


# ===============================================================
#   Extract bandwidth from path
# ===============================================================
def extract_bandwidth_from_path(path):
    m = re.search(r"band_([0-9.]+)", path)
    return m.group(1) if m else None


# ===============================================================
# ============================ MAIN =============================
# ===============================================================
def main():

    parameter_root = "parameter_test"
    out_root = os.path.join(parameter_root, "shear_vs_bandwidth")
    if rank == 0:
        os.makedirs(out_root, exist_ok=True)

    final_percentiles = [15]

    # -----------------------------------------------------------
    # Background catalog (FIXED)
    # -----------------------------------------------------------
    BG_data = os.path.join(
        parent_dir,
        "lhc_run_sims_zero_err_10",
        "run_1",
        "source_catalog_cutzl04.h5"
    )

    if not os.path.exists(BG_data):
        if rank == 0:
            print(f"[FATAL] Missing background file:\n  {BG_data}")
        return

    # -----------------------------------------------------------
    # Discover contracted ridge files
    # -----------------------------------------------------------
    contracted_files = find_contracted_files_parameter_test(parameter_root)

    if rank == 0:
        print(f"Found {len(contracted_files)} contracted ridge files.\n")

    skipped_missing_h5 = []
    skipped_existing_output = []

    # -----------------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------------
    for h5_file in contracted_files:

        try:
            if not os.path.exists(h5_file):
                skipped_missing_h5.append(h5_file)
                if rank == 0:
                    print(f"[missing] {h5_file}")
                continue

            # --------------------------------------------------
            # Extract bandwidth
            # --------------------------------------------------
            band_value = extract_bandwidth_from_path(h5_file)
            if band_value is None:
                skipped_missing_h5.append(h5_file)
                if rank == 0:
                    print(f"[skip] Could not extract bandwidth: {h5_file}")
                continue

            shear_csv    = os.path.join(out_root, f"shear_band_{band_value}.csv")
            filaments_h5 = os.path.join(out_root, f"filaments_band_{band_value}.h5")

            # --------------------------------------------------
            # Skip existing output
            # --------------------------------------------------
            if os.path.exists(shear_csv):
                skipped_existing_output.append(shear_csv)
                if rank == 0:
                    print(f"[skip] output exists: {shear_csv}")
                continue

            # --------------------------------------------------
            # Run shear computation
            # --------------------------------------------------
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
            if rank == 0:
                print(f"[ERROR] {h5_file} → {e}")

    # ===========================================================
    # FINAL SUMMARY
    # ===========================================================
    if rank == 0:
        print("\n================== FINAL SUMMARY ==================\n")

        if skipped_missing_h5:
            print(f"Missing contracted files ({len(skipped_missing_h5)}):")
            for f in skipped_missing_h5:
                print("  -", f)

        if skipped_existing_output:
            print(f"\nSkipped existing outputs ({len(skipped_existing_output)}):")
            for f in skipped_existing_output:
                print("  -", f)

        if not (skipped_missing_h5 or skipped_existing_output):
            print("No files were skipped.")

        print("\n==================================================")
        print("All shear calculations completed.")


# ===============================================================
# ENTRY POINT
# ===============================================================
if __name__ == "__main__":
    main()
