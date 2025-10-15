import os,sys
import h5py
import numpy as np
from ridge_analysis_tools import *




############################################################
############################################################

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())


############################################################
############################################################



# --- Application ---
base_sim_dir = "lhc_run_sims_with_err_10"
run_id = 1
bg_data_path = os.path.join(base_sim_dir, f"run_{run_id}", "source_catalog_cutzl04.h5")

output_dir = "simulation_ridges_comparative_analysis_test/noise"
os.makedirs(output_dir, exist_ok=True)

# Loop to generate 300 different noise realizations
num_realizations = 300
for i in range(num_realizations):
    # Use a different seed for each run to ensure different random rotations
    seed = 123 + i  
    output_file_path = os.path.join(output_dir, f"source_catalog_noise_{i:02d}.h5")
    
    # Call the function with the specific seed
    transform_background(bg_data_path, output_file_path, seed=seed)








