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
import healpy as hp
import matplotlib.pyplot as plt
import time
from ridge_analysis_tools import * 
import argparse
import datetime


#After execution the folder should look something like: 

"""
Cosmo_sim_ridges/
├── S8/
│   ├── run_1/
│   │   ├── band_0.1/
│   │   │   ├── checkpoints/
│   │   │   ├── Ridges_final_p15/
│   │   │   │   ├── S8_run_1_ridges_p15.h5
│   │   │   │   └── S8_run_1_ridges_p15_contracted.h5       ← NEW
│   │   │   ├── plots_by_final_percentile/
│   │   │   │   ├── S8_run_1_Ridges_plot_p15.png
│   │   │   │   └── S8_run_1_Ridges_plot_p15_contracted.png  ← NEW
│   │   ├── band_0.2/
│   │   │   └── ...
│   └── run_2/
│       └── ...
└── Om_fixed/
    ├── run_1/
    │   └── band_0.1/
    │       └── ...


"""



# ======================================================
# === CONFIGURATION ===
# ======================================================

mask_filename = "des-data/desy3_gold_mask.npy"   # mask file
base_root = "Cosmo_sim_ridges"  # main simulation folder
radius_arcmin = 4.0          # disk filter radius
min_coverage = 0.9           # fraction of mask pixels required
nside = 512                  # map resolution




# ======================================================
# === MAIN EXECUTION ===
# ======================================================

def main():
    parser = argparse.ArgumentParser(description="Ridge contraction pipeline.")
    parser.add_argument("--root", type=str, default="Cosmo_sim_ridges",
                        help="Root folder containing all categories (I set the default to zero noise cosmo sims).")
    parser.add_argument("--category", type=str, default=None,
                        help="list of categories to process (e.g. 'S8,S8_perp'). "
                             "If omitted, all categories under the root are processed.")
    parser.add_argument("--band", type=str, default=None,
                        help="specific band folder to process (currently we only have bandwidth 0.1').")
    parser.add_argument("--file", type=str, default=None,
                        help="specific ridge file to process.")
    args = parser.parse_args()

    root_dir = args.root
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    # === CATEGORY SELECTION ===
    if args.category:
        categories = [c.strip() for c in args.category.split(",")]
    else:
        categories = [c for c in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, c))]

    print(f"Categories to process: {categories}")

    #############################################################################
    # === SAFETY FLAGS ===

    CHECK_MISSING_INPUTS = True       # skip missing ridge files but record them
    CHECK_EXISTING_OUTPUTS = True     # skip already-processed (contracted) outputs

    missing_files = []  # collect missing input files for reporting later

    ############################################################################
    
    for category in categories:
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):
            print(f"[warnning] Category not found: {category}")
            continue

        run_folders = [r for r in os.listdir(category_path)
                       if r.startswith("run_") and os.path.isdir(os.path.join(category_path, r))]

        for run_folder in run_folders:
            run_path = os.path.join(category_path, run_folder)

            # === BAND SELECTION ===
            if args.band:
                band_folders = [args.band]
            else:
                band_folders = [b for b in os.listdir(run_path)
                                if b.startswith("band_") and os.path.isdir(os.path.join(run_path, b))]

            for band_folder in band_folders:
                band_path = os.path.join(run_path, band_folder)
                ridge_dir = os.path.join(band_path, "Ridges_final_p15")

                if not os.path.isdir(ridge_dir):
                    print(f"[skip] No ridge directory: {ridge_dir}")
                    continue

                # === RIDGE FILES ===
                if args.file:
                    ridge_files = [args.file]
                else:
                    ridge_files = [f for f in os.listdir(ridge_dir)
                                   if f.endswith(".h5") and "ridges" in f]

                for ridge_file in ridge_files:
                    ridge_path = os.path.join(ridge_dir, ridge_file)
                    if not os.path.isfile(ridge_path):
                        print(f"[skip] Ridge file not found: {ridge_path}")
                        continue
                    
                    # === SAFETY CHECK 1: Missing input file ===
                    if CHECK_MISSING_INPUTS and not os.path.isfile(ridge_path):
                        missing_files.append(ridge_path)
                        continue

                    # === SAFETY CHECK 2: Skip if contracted output already exists ===
                    if CHECK_EXISTING_OUTPUTS:
                        contracted_path = ridge_path.replace(".h5", "_contracted.h5")
                        if os.path.isfile(contracted_path):
                            print(f"[skip] Contracted file already exists: {contracted_path}")
                            continue
                    
                    
                    print(f"Processing: {ridge_path}")

                    # --- ridge contraction routine ---
                    try:
                        contract_ridge_file(ridge_path, category, run_folder, band_folder)
                    except Exception as e:
                        print(f"[error] Failed to process {ridge_path}: {e}")
                        continue


    # === FINAL REPORT FOR MISSING INPUTS ===
    if CHECK_MISSING_INPUTS and missing_files:
        print("\n=== Summary: Missing Ridge Files ===")
        for f in missing_files:
            print(f" - {f}")
        print("====================================\n")
    else:
        print("\nAll existing ridge files processed successfully.\n")

if __name__ == "__main__":
    main()
