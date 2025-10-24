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


filament_dir = f"simulation_ridges_comparative_analysis_debug/normal/band_0.1/shear_test_data_{run_id}"  
os.makedirs(filament_dir, exist_ok=True)

final_percentiles = [15] #[0, 10, 25, 40, 50, 60, 75, 85, 90, 95]
for fp in final_percentiles:
    if comm is None or comm.rank == 0:
        print(f"[rank 0] Processing filaments for final_percentile={fp}")
        h5_file = f"simulation_ridges_comparative_analysis_debug/normal/band_0.1/contracted_Ridges_final_p15/normal_run_1_ridges_p15_contracted.h5"
        
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
    #shear_flip_csv = os.path.join(filament_dir, f"shear_p{fp:02d}_flipG1.csv")
    
    # Run with normal signs
    process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv, k=1, num_bins=20, comm=comm,
                           flip_g1=False, flip_g2=False, background_type='sim', nside_coverage=32,
                           min_distance_arcmin=1.0, max_distance_arcmin=60.0)
#    process_shear_sims(filament_h5, BG_data, output_shear_file = shear_flip_csv, k=1, num_bins=20, comm=comm,
#                       flip_g1=True, flip_g2=False, background_type='sim', nside_coverage=32,
#                       min_distance_arcmin=1.0, max_distance_arcmin=60.0)