import os
import sys

# ==============================================================
# PATH SETUP
# ==============================================================

# Directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# One level up 
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Make parent importable 
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import h5py
from ridge_analysis_tools import * 


# ==============================================================
# DISCOVER RUNS
# ==============================================================

def discover_run_dirs(base_root, nz_dir="lsst10_nz", err_modes=("noise", "zero_err")):
    """
    Return absolute paths to run_*:  contain source_catalog_0.npy.

    Expected layout:
      base_root/
        noise/
          lsst10_nz/
            run_1/
              source_catalog_0.npy
        zero_err/
          lsst10_nz/
            run_1/
              source_catalog_0.npy
    """
    if not os.path.exists(base_root):
        raise FileNotFoundError(f"Base path does not exist: {base_root}")

    run_dirs = []
    for err_mode in err_modes:
        root = os.path.join(base_root, err_mode, nz_dir)
        if not os.path.isdir(root):
            continue

        for d in sorted(os.listdir(root)):
            run_path = os.path.join(root, d)
            if not os.path.isdir(run_path):
                continue

            npy_file = os.path.join(run_path, "source_catalog_0.npy")
            if os.path.exists(npy_file):
                run_dirs.append(run_path)

    return run_dirs


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":

    # Your simulation folder lives at:
    #   ~/WL_Mehraveh/ridge-sims/lhc_run_hybrid_DES_lSST10_sims
    #
    # With current_dir = .../ridge-sims/<this_script_folder>
    # parent_dir points to .../ridge-sims
    sim_root_name = "lhc_run_hybrid_DES_lSST10_sims"
    base_root = os.path.join(parent_dir, sim_root_name)

    print(f"[INFO] Scanning base: {base_root}")

    run_dirs = discover_run_dirs(base_root, nz_dir="lsst10_nz")
    
    for rd in run_dirs:
        print("\n=== Processing run directory ===")
        print(rd)

        # Apply your background cut / conversion
        convert_all_backgrounds(rd, z_cut=0.7)

    print("\n[INFO] Done.")
