import os
import pandas as pd
import numpy as np
import h5py
import time
from ridge_analysis_tools import *
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt  

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

# Load background data for shear
base_sim_dir = "lhc_run_sims"
run_id = 1
BG_data = os.path.join(base_sim_dir, f"run_{run_id}", "source_catalog_cutzl04.h5")


#filament_dir = "example/filaments"
#filament_dir = "example_zl04/filaments"
filament_dir = "example_zl04_mesh5e5/filaments/shrinked_filaments" # added sub-directory for filaments created from shrinked ridges
os.makedirs(filament_dir, exist_ok=True)

final_percentiles = [15] #[0, 10, 25, 40, 50, 60, 75, 85, 90, 95]
for fp in final_percentiles:
    if comm is None or comm.rank == 0:
        print(f"[rank 0] Processing filaments for final_percentile={fp}")
        #h5_file = f"example_zl04_mesh5e5/Ridges_final_p{fp:02d}/ridges_p{fp:02d}.h5"
        h5_file = f"example_zl04_mesh5e5/shrinked_ridges/ridges_p{fp:02d}_shrinked.h5" # the shrinked ridge file 
        #h5_file = f"example/Ridges_final_p{fp:02d}/ridges_p{fp:02d}.h5"
        with h5py.File(h5_file, 'r') as f:
            Ridges = f["ridges"][:]

        mst = build_mst(Ridges)
        branch_points = detect_branch_points(mst)
        filament_segments = split_mst_at_branches(mst, branch_points)
        filament_labels = segment_filaments_with_dbscan(Ridges, filament_segments)

        filament_h5 = os.path.join(filament_dir, f"filaments_p{fp:02d}.h5")
        save_filaments_to_hdf5(Ridges, filament_labels, filament_h5)

    if comm is not None:
        comm.Barrier()

    # Shear processing (all ranks)
    filament_h5 = os.path.join(filament_dir, f"filaments_p{fp:02d}.h5")
    shear_csv = os.path.join(filament_dir, f"shear_p{fp:02d}.csv")

    
    # Run with normal signs
    process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv, background_type='sim')
    

    # ===== Test G1/G2 sign flips =======
	
    process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv.replace(".csv", "_flipG1.csv"), flip_g1=True, background_type='sim')
    #process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv.replace(".csv", "_flipG2.csv"), flip_g2=True, background_type='sim')
    #process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv.replace(".csv", "_flipG1G2.csv"), flip_g1=True, flip_g2=True, background_type='sim')
