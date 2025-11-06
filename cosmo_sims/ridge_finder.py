import os, sys

# directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Change working dir to script dir
os.chdir(current_dir)

# Add both the current and parent directories to search path
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)



import numpy as np
import h5py
import dredge_scms
import healpy
import matplotlib.pyplot as plt
from ridge_analysis_tools import * 
from mpi4py.MPI import COMM_WORLD






def main():
    base_sim_dir = "lhc_cosmo_sims_zero_err"
    output_base = "Cosmo_sim_ridges"
    os.makedirs(output_base, exist_ok=True)

    categories = ["S8", "S8_perp", "Om_fixed", "sigma8_fixed"]
    num_runs = 10
    bandwidth = 0.1

    for category in categories:
        for run_idx in range(1, num_runs + 1):
            if COMM_WORLD.rank == 0:
                print(f"\n[rank 0] Starting ridge extraction for {category}/run_{run_idx}")

            sim_dir = os.path.join(base_sim_dir, category)
            base_label = category  # for filenames inside the function

            # New directory structure
            home_dir = os.path.join(output_base, category, f"run_{run_idx}", f"band_{bandwidth:.1f}")
            os.makedirs(home_dir, exist_ok=True)

            run_filament_pipeline(
                bandwidth=bandwidth,
                base_sim_dir=sim_dir,
                run_ids=[run_idx],
                base_label=base_label,
                home_dir=home_dir
            )

    if COMM_WORLD.rank == 0:
        print("\nAll ridge extraction runs complete.")


if __name__ == "__main__":
    main()




