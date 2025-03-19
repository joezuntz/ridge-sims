####################### IMPORT lIBRARIES 

import os
import pandas as pd
import numpy as np
import h5py
#from dredge_mod import filaments
import importlib
from tools import *
from sklearn.neighbors import KernelDensity

################## Load the Foreground and background data


BG_data= "BG_data_1x1.hdf5"  # Test set


FG_data = "FG_data.hdf5" # Test set

with h5py.File(FG_data, "r") as hdf_in:
    ra = hdf_in["filtered_catalog/ra"][:]
    dec = hdf_in["filtered_catalog/dec"][:]

# Stack RA and DEC into a two-column NumPy array
ra_dec_array = np.column_stack((ra, dec))

# Remove rows with NaN or inf
valid_rows = np.isfinite(ra_dec_array).all(axis=1)
clean_coordinates = ra_dec_array[valid_rows]

# Create necessary directories
os.makedirs("filament_outputs", exist_ok=True)
os.makedirs("filament_variation_outputs", exist_ok=True)
os.makedirs("shear_outputs", exist_ok=True)
os.makedirs("shear_variation_outputs", exist_ok=True)



# Ask the user for input configurations
use_default_filament = input("Use default filament parameters? (yes/no): ").strip().lower()

# Choose the appropriate module 
dredge_module_name = "dredge_mod" if use_default_filament == "yes" else "dredge_mod2"
dredge_module = importlib.import_module(dredge_module_name)

# Import the `filaments` function from the selected module
filaments = dredge_module.filaments


if use_default_filament == "no":
    param_variation = input("Do you want to use a single configuration (single) or loop through all variations (loop)? ").strip().lower()

    if param_variation == "single":
        # Get user input for a single run
        neighbors = int(input("Enter number of neighbors: ").strip())
        bandwidth = input("Enter bandwidth (or None): ").strip()
        bandwidth = None if bandwidth.lower() == "none" else float(bandwidth)
        convergence = float(input("Enter convergence threshold: ").strip())
        percentage = input("Enter percentage (or None): ").strip()
        percentage = None if percentage.lower() == "none" else float(percentage)
        mesh_size = input("Enter mesh_size (or None): ").strip()
        mesh_size = None if mesh_size.lower() == "none" else int(mesh_size)

        print("Running single filament detection...")
        Ridges = filaments(clean_coordinates, neighbors=neighbors, bandwidth=bandwidth, 
                           convergence=convergence, percentage=percentage, distance='haversine', 
                           n_process=0, mesh_size=mesh_size)
        
        output_file = f"filament_outputs/Ridges_output_nb{neighbors}_bw{bandwidth}_conv{convergence}_perc{percentage}_mesh{mesh_size}.hdf5"
        save_to_hdf5(Ridges, output_file)
        print(f"Saved filament data to {output_file}")
        
        # Process MST and DBSCAN segmentation
        mst = build_mst(Ridges)
        branch_points = detect_branch_points(mst)
        filament_segments = split_mst_at_branches(mst, branch_points)
        filament_labels = segment_filaments_with_dbscan(Ridges, filament_segments)
        save_filaments_to_hdf5(Ridges, filament_labels, output_file.replace("Ridges_output", "Filament_labels"))

        # Process shear data
        filament_file = output_file.replace("Ridges_output", "Filament_labels")
        shear_output_file = "shear_outputs/observed_shear_transformed_weighted.csv"
        process_shear(filament_file, BG_data, output_shear_file=shear_output_file)
        print(f"Shear processing completed. Output saved: {shear_output_file}")

    else:
        param_to_vary = input("Which parameter would you like to vary? (neighbors/bandwidth/convergence/percentage/mesh_size): ").strip().lower()
        start_value = float(input(f"Enter the start value for {param_to_vary}: ").strip())
        end_value = float(input(f"Enter the end value for {param_to_vary}: ").strip())
        steps = int(input("Enter the number of steps: ").strip())
        varied_values = np.linspace(start_value, end_value, steps).tolist()

        for i, varied_value in enumerate(varied_values):
            print(f"Running variation {i+1}/{steps} with {param_to_vary} = {varied_value}...")

            neighbors, bandwidth, convergence, percentage, mesh_size = 10, None, 0.005, None, None
            if param_to_vary == "neighbors":
                neighbors = int(varied_value)
            elif param_to_vary == "bandwidth":
                bandwidth = float(varied_value)
            elif param_to_vary == "convergence":
                convergence = varied_value
            elif param_to_vary == "percentage":
                percentage = varied_value
            elif param_to_vary == "mesh_size":
                mesh_size = varied_value

            Ridges = filaments(clean_coordinates, neighbors=neighbors, bandwidth=bandwidth, 
                               convergence=convergence, percentage=percentage, distance='haversine', 
                               n_process=0, mesh_size=mesh_size)

            output_file = f"filament_variation_outputs/Ridges_output_{i+1}_{param_to_vary}{varied_value}.hdf5"
            save_to_hdf5(Ridges, output_file)
            print(f"Saved filament data to {output_file}")

            # Process MST and DBSCAN segmentation
            mst = build_mst(Ridges)
            branch_points = detect_branch_points(mst)
            filament_segments = split_mst_at_branches(mst, branch_points)
            filament_labels = segment_filaments_with_dbscan(Ridges, filament_segments)
            save_filaments_to_hdf5(Ridges, filament_labels, output_file.replace("Ridges_output", "Filament_labels"))

            # Process shear data
            filament_file = output_file.replace("Ridges_output", "Filament_labels")
            shear_output_file = f"shear_variation_outputs/observed_shear_transformed_weighted_{param_to_vary}_{i+1}.csv"
            process_shear(filament_file, BG_data, output_shear_file=shear_output_file)
            print(f"Shear processing for variation {i+1} completed. Output saved: {shear_output_file}")
else:
    print("Running default filament detection...")
    Ridges = filaments(clean_coordinates, neighbors=10, bandwidth=None, convergence=0.005, percentage=None, distance='haversine', n_process=0, mesh_size=None)
    output_file = "filament_outputs/Ridges_output_default.hdf5"
    save_to_hdf5(Ridges, output_file)
    print(f"Saved ridges data to {output_file}")

    # Process MST and DBSCAN segmentation
    mst = build_mst(Ridges)
    branch_points = detect_branch_points(mst)
    filament_segments = split_mst_at_branches(mst, branch_points)
    filament_labels = segment_filaments_with_dbscan(Ridges, filament_segments)
    save_filaments_to_hdf5(Ridges, filament_labels, "filament_outputs/Filament_labels_default.hdf5")
    print("saved segmented filament results")
    # Process shear data
    filament_file= "filament_outputs/Filament_labels_default.hdf5"
    shear_output_file = "shear_outputs/observed_shear_transformed_weighted_default.csv"
    process_shear(filament_file, BG_data, output_shear_file=shear_output_file)
    print(f"Shear processing completed. Output saved: {shear_output_file}")

print("All filament variations completed successfully!")
