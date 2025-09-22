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

# noise from random shear permutation
noise_dir = "example_zl04_mesh5e5/noise"
noise = os.path.join(noise_dir, "source_catalog_noise.h5")

# filament output
filament_dir = "example_zl04_mesh5e5/filaments"
os.makedirs(filament_dir, exist_ok=True)

final_percentiles = [15]   # can loop over multiple percentiles if needed
for fp in final_percentiles:
    if comm is None or comm.rank == 0:
        print(f"[rank 0] Processing filaments for final_percentile={fp}")
        h5_file = f"example_zl04_mesh5e5/Ridges_final_p{fp:02d}/ridges_p{fp:02d}.h5"
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

    filament_h5 = os.path.join(filament_dir, f"filaments_p{fp:02d}.h5")
    
    # Define file paths for shear outputs
    shear_csv = os.path.join(filament_dir, f"shear_p{fp:02d}.csv")
    shear_flip_csv = os.path.join(filament_dir, f"shear_p{fp:02d}_flipG1.csv")
    shear_noise_csv = os.path.join(noise_dir, f"shear_noise_p{fp:02d}.csv")
    shear_noise_flip_csv = os.path.join(noise_dir, f"shear_noise_p{fp:02d}_flipG1.csv")
    
    # --- Run with normal signs (signal and noise) ---
    process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv, background_type='sim')
    process_shear_sims(filament_h5, noise, output_shear_file=shear_noise_csv, background_type='sim')

    # --- Run with flipped signs (signal and noise) ---
    process_shear_sims(filament_h5, BG_data, output_shear_file=shear_flip_csv,
                       flip_g1=True, background_type='sim')
    process_shear_sims(filament_h5, noise, output_shear_file=shear_noise_flip_csv,
                       flip_g1=True, background_type='sim')

    # === Subtract noise from normal signal ===
    if comm is None or comm.rank == 0:
        print(f"Subtracting noise from signal for final_percentile={fp}")
        
        shear_data = np.loadtxt(shear_csv, delimiter=",", skiprows=1)
        noise_data = np.loadtxt(shear_noise_csv, delimiter=",", skiprows=1)

        g_plus_subtracted = shear_data[:, 2] - noise_data[:, 2]
        g_cross_subtracted = shear_data[:, 3] - noise_data[:, 3]

        subtracted_data = np.column_stack((
            shear_data[:, 0],  # Bin_Center
            shear_data[:, 1],  # Weighted_Real_Distance
            g_plus_subtracted,
            g_cross_subtracted,
            shear_data[:, 4],  # Counts
            shear_data[:, 5]   # bin_weight
        ))

        subtracted_output_file = os.path.join(
            filament_dir, f"shear_p{fp:02d}_shear-randomshear.csv"
        )
        np.savetxt(
            subtracted_output_file,
            subtracted_data,
            delimiter=",",
            header="Bin_Center,Weighted_Real_Distance,Weighted_g_plus_subtracted,Weighted_g_cross_subtracted,Counts,bin_weight",
            comments=""
        )
        print(f"Subtracted shear profiles saved to {subtracted_output_file}")

    # === Subtract noise from flipped signal ===
    if comm is None or comm.rank == 0:
        print(f"Subtracting flipped noise from flipped signal for final_percentile={fp}")
        
        shear_data_flip = np.loadtxt(shear_flip_csv, delimiter=",", skiprows=1)
        noise_data_flip = np.loadtxt(shear_noise_flip_csv, delimiter=",", skiprows=1)

        g_plus_subtracted_flip = shear_data_flip[:, 2] - noise_data_flip[:, 2]
        g_cross_subtracted_flip = shear_data_flip[:, 3] - noise_data_flip[:, 3]

        subtracted_data_flip = np.column_stack((
            shear_data_flip[:, 0],  # Bin_Center
            shear_data_flip[:, 1],  # Weighted_Real_Distance
            g_plus_subtracted_flip,
            g_cross_subtracted_flip,
            shear_data_flip[:, 4],  # Counts
            shear_data_flip[:, 5]   # bin_weight
        ))

        subtracted_output_file_flip = os.path.join(
            filament_dir, f"shear_p{fp:02d}_flipG1_shear-randomshear.csv"
        )
        np.savetxt(
            subtracted_output_file_flip,
            subtracted_data_flip,
            delimiter=",",
            header="Bin_Center,Weighted_Real_Distance,Weighted_g_plus_subtracted,Weighted_g_cross_subtracted,Counts,bin_weight",
            comments=""
        )
        print(f"Subtracted flipped shear profiles saved to {subtracted_output_file_flip}")
