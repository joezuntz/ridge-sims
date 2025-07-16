import os
import pandas as pd
import numpy as np
import h5py
import importlib
#from tools import *
from ridge_analysis.tools import * 
from sklearn.neighbors import KernelDensity
import dredge_mod  # Assuming 'dredge_mod.py' is in the same directory

# Define the directory containing the little squares
little_runs_dir = "little_runs_adaptive"  
output_base_dir = "convergence_test"

# Define the convergence values to loop over
start = 1e-7  # Adjusted start value 
stop = 1e-5
num = 5
convergence_values = np.logspace(np.log10(start), np.log10(stop), num) # Using logspace for better distribution (?)

# Create the base output directory if it doesn't exist
os.makedirs(output_base_dir, exist_ok=True)

# Get a list of all the little square files
little_square_files = [f for f in os.listdir(little_runs_dir) if f.endswith(".npy") and "little_run_" in f]

if not little_square_files:
    print(f"No little square files found in '{little_runs_dir}'. Please ensure they exist.")
else:
    for square_file in little_square_files:
        run_id = square_file.split("_")[2].split(".")[0]  # Extract run ID from filename 
        little_square_path = os.path.join(little_runs_dir, square_file)
        little_square_data = np.load(little_square_path)

        # Reshape if it's a 5x5x2 array back to (N, 2)
        if little_square_data.ndim == 3:
            clean_coordinates = little_square_data.reshape(-1, 2)
        else:
            clean_coordinates = little_square_data

        # Remove rows with NaN or inf
        valid_rows = np.isfinite(clean_coordinates).all(axis=1)
        clean_coordinates = clean_coordinates[valid_rows]

        if clean_coordinates.shape[0] < 20: # Ensure enough points exists for processing
            print(f"Skipping {square_file}: Not enough valid coordinates (< 20).")
            continue

        for convergence in convergence_values:
            # Create a subdirectory for the current convergence value and run
            convergence_str = f"{convergence:.0e}".replace("+0", "").replace("-0", "-")
            output_dir = os.path.join(output_base_dir, f"run_{run_id}", f"convergence_{convergence_str}")
            os.makedirs(output_dir, exist_ok=True)

            convergence = float(convergence)
            print(f"\nProcessing {square_file} with convergence = {convergence}")

            try:
                ridges = dredge_mod.filaments(
                    clean_coordinates,
                    bandwidth=0.035,  # adjust this later if needed
                    convergence=convergence,
                    percentage=None,
                    distance='haversine',
                    n_neighbors=20,
                    n_process=1,
                    plot_dir=os.path.join(output_dir, 'plots'),
                    mesh_size=None
                )

                output_file = os.path.join(output_dir, "Ridges_output.hdf5")
                save_to_hdf5(ridges, output_file)
                print(f"  Saved ridges data to {output_file}")

                # Process MST and DBSCAN segmentation
                mst = build_mst(ridges)
                branch_points = detect_branch_points(mst)
                filament_segments = split_mst_at_branches(mst, branch_points)
                filament_labels = segment_filaments_with_dbscan(ridges, filament_segments)
                filament_labels_output_file = os.path.join(output_dir, f"Filament_labels_convergence_{convergence_str}.hdf5")
                save_filaments_to_hdf5(ridges, filament_labels, filament_labels_output_file)
                print(f"  Saved segmented filament results to {filament_labels_output_file}")

            except Exception as e:
                print(f"  Error processing {square_file} with convergence {convergence}: {e}")

    print("\nFinished processing all little squares for all convergence values.")