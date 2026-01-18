
import os,sys
# Path setup --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))      
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))  

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

os.chdir(current_dir)



import h5py
import numpy as np
from ridge_analysis_tools import *



# --- Application ---
base_sim_dir =os.path.join(parent_dir, "lhc_run_sims")
run_id = 1
bg_data_path = os.path.join(base_sim_dir, f"run_{run_id}", "source_catalog_cutzl04.h5")

output_dir = os.path.join(current_dir,"DES_sim/noise")
os.makedirs(output_dir, exist_ok=True)

# Loop to generate 300 different noise realizations
num_realizations = 300
for i in range(num_realizations):
    # Use a different seed for each run to ensure different random rotations
    seed = 123 + i  
    output_file_path = os.path.join(output_dir, f"source_catalog_noise_{i:02d}.h5")
    
    # Call the function with the specific seed
    transform_background(bg_data_path, output_file_path, seed=seed)


