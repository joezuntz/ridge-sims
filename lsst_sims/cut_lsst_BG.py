import os, sys

# Directory of this script (cosmo_sims)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Make parent importable 
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)



import numpy as np
import h5py
from ridge_analysis_tools import *

####################################
# === discover LSST BG directories
####################################

def discover_run_dirs(base_root):
    """
    Find all subdirectories of form lsst_*/run_*/ that contain
    source_catalog_0.npy. Returns a list of absolute paths.
    """
    run_dirs = []

    if not os.path.exists(base_root):
        raise FileNotFoundError(f"Base path does not exist: {base_root}")

    for lsst_dir in sorted(os.listdir(base_root)):
        lsst_path = os.path.join(base_root, lsst_dir)
        if not os.path.isdir(lsst_path):
            continue

        for run_dir in sorted(os.listdir(lsst_path)):
            run_path = os.path.join(lsst_path, run_dir)
            if not os.path.isdir(run_path):
                continue

            npy_file = os.path.join(run_path, "source_catalog_0.npy")
            if os.path.exists(npy_file):
                run_dirs.append(run_path)

    return run_dirs


############################################################
# === Main: apply BG conversion to all LSST runs
############################################################

if __name__ == "__main__":

    roots = [
        "lhc_run_lsst_sims",
        "lhc_run_lsst_sim_zero_err"
    ]

    for root in roots:
        # go to parent directory
        base_root = os.path.join(parent_dir, root)

        print(f"[INFO] Scanning base: {base_root}")

        try:
            run_dirs = discover_run_dirs(base_root)
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")
            continue

        print(f"[INFO] Found {len(run_dirs)} run directories")

        for rd in run_dirs:
            print("\n=== Processing run directory ===")
            print(rd)
            convert_all_backgrounds(rd)