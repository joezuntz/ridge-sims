import os
import pandas as pd
import numpy as np
import h5py
import dredge_mod
from dredge_mod import filaments
from tools import *


BG_data_clean = "BG_data.hdf5"

# Create necessary directories
os.makedirs("Simulation_background_outputs", exist_ok=True)
os.makedirs("Simulation_foreground_outputs", exist_ok=True)
os.makedirs("Simulation_shear_outputs", exist_ok=True)

# User options (only for simulation mode)
perform_background = input("Perform background transformation? (yes/no): ").strip().lower() == "yes"
perform_foreground = input("Perform foreground generation? (yes/no): ").strip().lower() == "yes"
use_default_filament = input("Use default filament parameters? (yes/no): ").strip().lower() == "yes"
num_simulations = int(input("Enter the number of simulations: ").strip())

if not use_default_filament:
    neighbors = int(input("Enter number of neighbors: ").strip())
    bandwidth = input("Enter bandwidth (or None): ").strip()
    bandwidth = None if bandwidth.lower() == "none" else float(bandwidth)
    convergence = float(input("Enter convergence threshold: ").strip())
    percentage = input("Enter percentage (or None): ").strip()
    percentage = None if percentage.lower() == "none" else float(percentage)
    mesh_size = input("Enter mesh size (or None): ").strip()
    mesh_size = None if mesh_size.lower() == "none" else float(mesh_size)

# Prompt for foreground generation parameters only once
if perform_foreground:
    ra_min = float(input("Enter RA min (default 32): ") or 32)
    ra_max = float(input("Enter RA max (default 33): ") or 33)
    dec_min = float(input("Enter DEC min (default -44): ") or -44)
    dec_max = float(input("Enter DEC max (default -43): ") or -43)
    num_points = int(input("Enter the number of points (default 300): ") or 300)

print(f"Running {num_simulations} simulations...")

for i in range(num_simulations):
    print(f"Running simulation {i+1}/{num_simulations}...")

    # Create filename suffix based on user choices
    bg_flag = "_BG" if perform_background else ""
    fg_flag = "_FG" if perform_foreground else ""

    # Perform background transformation if selected
    if perform_background:
        background_hdf5_file = f"Simulation_background_outputs/random_background_{i+1}{bg_flag}.h5"
        transform_background(BG_data_clean, background_hdf5_file, i)

    # Perform foreground generation if selected
    if perform_foreground:
        foreground_hdf5_file = f"Simulation_foreground_outputs/foreground_output_simulation_{i+1}{fg_flag}.h5"
        foreground_points = generate_foreground(i, ra_min, ra_max, dec_min, dec_max, num_points, foreground_hdf5_file)
        #print("Foreground points shape:", foreground_points.shape)

    # Determine output file name based on filament parameters and processing flags
    if use_default_filament:
        output_file = f"Simulation_foreground_outputs/Ridges_output_simulation_{i+1}_default{bg_flag}{fg_flag}.hdf5"
    else:
        output_file = (
            f"Simulation_foreground_outputs/Ridges_output_simulation_{i+1}_"
            f"n{num_points}_nb{neighbors}_bw{bandwidth}_conv{convergence}_perc{percentage}_mesh{mesh_size}"
            f"{bg_flag}{fg_flag}.hdf5"
        )

    # If foreground generation was skipped, load existing data
    if not perform_foreground:
        FG_data = "FG_data.hdf5" # Test set
        with h5py.File(FG_data, "r") as hdf_in:
            ra = hdf_in["filtered_catalog/ra"][:]
            dec = hdf_in["filtered_catalog/dec"][:]

        # Stack RA and DEC into a two-column NumPy array
        ra_dec_array = np.column_stack((ra, dec))

        # Remove rows with NaN or inf
        valid_rows = np.isfinite(ra_dec_array).all(axis=1)
        clean_coordinates = ra_dec_array[valid_rows]

        FilamentData = filaments(
            clean_coordinates,
            neighbors=10 if use_default_filament else neighbors,
            bandwidth=None if use_default_filament else bandwidth,
            convergence=0.005 if use_default_filament else convergence,
            percentage=None if use_default_filament else percentage,
            mesh_size=None if use_default_filament else mesh_size
        )

        save_to_hdf5(FilamentData, output_file)

    else:
        FilamentData = filaments(
            foreground_points,
            neighbors=10 if use_default_filament else neighbors,
            bandwidth=None if use_default_filament else bandwidth,
            convergence=0.005 if use_default_filament else convergence,
            percentage=None if use_default_filament else percentage,
            mesh_size=None if use_default_filament else mesh_size
        )

        save_to_hdf5(FilamentData, output_file)

    
    # Process MST, detect branch points, split MST at branches
    mst = build_mst(FilamentData)
    
    branch_points = detect_branch_points(mst)
    filament_segments = split_mst_at_branches(mst, branch_points)
    
    # Segment filaments using DBSCAN and save the segmented results
    filament_labels = segment_filaments_with_dbscan(FilamentData, filament_segments)
    
    simulated_filament_file = f"Simulation_foreground_outputs/Simulated_Filaments_Segments_{i+1}{bg_flag}{fg_flag}.hdf5"
    save_filaments_to_hdf5(FilamentData, filament_labels, simulated_filament_file)

    # Process shear data and save the results
    shear_output_file = f"Simulation_shear_outputs/simulated_shear_{i+1}{bg_flag}{fg_flag}.csv"
    process_shear1(
    simulated_filament_file,
    BG_data_clean if not perform_background else background_hdf5_file,
    output_shear_file=shear_output_file
    )
    print(f"Shear processing complete. Output saved: {shear_output_file}")

print("All simulations completed successfully!")
