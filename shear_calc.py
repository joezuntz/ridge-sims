import os
import pandas as pd
import numpy as np
import h5py
import time
from ridge_analysis_tools import *
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from mpi4py import MPI
comm = MPI.COMM_WORLD

def main():
    base_sim_dir = "lhc_run_sims_50"
    num_runs = 8
    final_percentile = 15
    
    for run_id in range(1, num_runs + 1):
        if comm.rank == 0:
            print(f"--- Processing Run {run_id} for Shear ---")
            
        # Define directories and file paths for the current run
        output_base_dir = f"example30_band0.4/8test/run_{run_id}"
        filament_dir = os.path.join(output_base_dir, "filaments_filtered")
        os.makedirs(filament_dir, exist_ok=True)

        BG_data = os.path.join(base_sim_dir, f"run_{run_id}", "source_catalog_cutzl04.h5")
        
        h5_file = os.path.join(output_base_dir, f"ridges_filtered/ridges_p{final_percentile:02d}_filtered.h5")
        
        with h5py.File(h5_file, 'r') as f:
            Ridges = f["ridges"][:]

        mst = build_mst(Ridges)
        branch_points = detect_branch_points(mst)
        filament_segments = split_mst_at_branches(mst, branch_points)
        filament_labels = segment_filaments_with_dbscan(Ridges, filament_segments)
        
        filament_h5 = os.path.join(filament_dir, f"filaments_p{final_percentile:02d}.h5")
        save_filaments_to_hdf5(Ridges, filament_labels, filament_h5)
        
        # Use the segmented filament file 
        shear_csv = os.path.join(filament_dir, f"shear_p{final_percentile:02d}.csv")

        # Process filaments and calculate shear
        process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv, background_type='sim')
        
        # ===== Test G1/G2 sign flips =======
	
        process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv.replace(".csv", "_flipG1.csv"), flip_g1=True, background_type='sim')
        #process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv.replace(".csv", "_flipG2.csv"), flip_g2=True, background_type='sim')
        #process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv.replace(".csv", "_flipG1G2.csv"), flip_g1=True, flip_g2=True, background_type='sim')

if __name__ == "__main__":
    main()