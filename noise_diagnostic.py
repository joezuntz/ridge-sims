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

# === Input paths ===
base_sim_dir = "lhc_run_sims"
run_id = 1
BG_data = os.path.join(base_sim_dir, f"run_{run_id}", "source_catalog_cutzl04.h5")

# Noise realizations directory
noise_dir = "example_zl04_mesh5e5/noise"
shear_noise_dir = os.path.join(noise_dir, "shear")
os.makedirs(shear_noise_dir, exist_ok=True)

# Filament output directory
filament_dir = "example_zl04_mesh5e5/filaments"
os.makedirs(filament_dir, exist_ok=True)

# === Run for chosen percentiles ===
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

    # === File paths for signal shear ===
    shear_csv = os.path.join(filament_dir, f"shear_p{fp:02d}.csv")
    shear_flip_csv = os.path.join(filament_dir, f"shear_p{fp:02d}_flipG1.csv")

    # --- Run with signal ---
    process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv, background_type='sim')
    process_shear_sims(filament_h5, BG_data, output_shear_file=shear_flip_csv,
                       flip_g1=True, background_type='sim')

    # === Loop over noise realizations ===
    noise_files = sorted(
    [f for f in os.listdir(noise_dir) if f.startswith("source_catalog_noise_") and f.endswith(".h5")],
    key=lambda x: int(x.split("_")[-1].replace(".h5","")))

    all_noise_profiles = []
    all_noise_flip_profiles = []

    for nf in noise_files:
        realization_id = nf.split("_")[-1].replace(".h5", "")  # file naming sytem : "00", "01"
        noise_file = os.path.join(noise_dir, nf)

        # Output shear files per realization
        shear_noise_csv_i = os.path.join(shear_noise_dir, f"shear_noise_p{fp:02d}_{realization_id}.csv")
        shear_noise_flip_csv_i = os.path.join(shear_noise_dir, f"shear_noise_p{fp:02d}_{realization_id}_flipG1.csv")

        # Compute shear for this noise realization
        process_shear_sims(filament_h5, noise_file, output_shear_file=shear_noise_csv_i, background_type='sim')
        process_shear_sims(filament_h5, noise_file, output_shear_file=shear_noise_flip_csv_i,
                           flip_g1=True, background_type='sim')

        # Load into memory 
        all_noise_profiles.append(np.loadtxt(shear_noise_csv_i, delimiter=",", skiprows=1))
        all_noise_flip_profiles.append(np.loadtxt(shear_noise_flip_csv_i, delimiter=",", skiprows=1))

    # Convert to arrays (N_realizations, N_bins, N_columns)
    all_noise_profiles = np.array(all_noise_profiles)
    all_noise_flip_profiles = np.array(all_noise_flip_profiles)

    # Mean across realizations
    mean_noise = np.mean(all_noise_profiles, axis=0)
    mean_noise_flip = np.mean(all_noise_flip_profiles, axis=0)

    # === Subtract mean noise from signal ===
    if comm is None or comm.rank == 0:
        print(f"Subtracting mean noise from signal")

        shear_data = np.loadtxt(shear_csv, delimiter=",", skiprows=1)
        shear_data_flip = np.loadtxt(shear_flip_csv, delimiter=",", skiprows=1)

        g_plus_subtracted = shear_data[:, 2] - mean_noise[:, 2]
        g_cross_subtracted = shear_data[:, 3] - mean_noise[:, 3]

        g_plus_subtracted_flip = shear_data_flip[:, 2] - mean_noise_flip[:, 2]
        g_cross_subtracted_flip = shear_data_flip[:, 3] - mean_noise_flip[:, 3]

        # Save subtracted signal
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
        print(f"Saved mean-noise-subtracted shear to {subtracted_output_file}")

        # Save flipped subtracted signal
        subtracted_data_flip = np.column_stack((
            shear_data_flip[:, 0],
            shear_data_flip[:, 1],
            g_plus_subtracted_flip,
            g_cross_subtracted_flip,
            shear_data_flip[:, 4],
            shear_data_flip[:, 5]
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
        print(f"Saved mean-noise-subtracted flipped shear to {subtracted_output_file_flip}")
