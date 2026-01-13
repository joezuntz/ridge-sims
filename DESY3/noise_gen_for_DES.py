import os
import sys
# ==============================================================
# PATH SETUP
# ==============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

os.chdir(current_dir)

import dredge_scms
#from ridge_analysis_tools import *





import numpy as np
import h5py
import re
from ridge_analysis_tools import *



# ------------------------------------------------------------
# HELPER FUNCTION 
# ------------------------------------------------------------

def transform_DES_background(background_file, output_hdf5_file, seed):
    """Applies random rotation to shear values and saves the transformed background to an HDF5 file."""
    
    np.random.seed(seed)

    with h5py.File(background_file, "r") as file:
        bg_ra = file["ra"][:]
        bg_dec = file["dec"][:]
        g1_values = file["g1"][:]
        g2_values = file["g2"][:]

        has_weight = "weight" in file
        weights = file["weight"][:] if has_weight else np.ones_like(bg_ra)

    # Random rotation
    psi = np.random.uniform(0, 2 * np.pi, size=len(bg_ra))
    cos_2psi = np.cos(2 * psi)
    sin_2psi = np.sin(2 * psi)

    g1_transformed = cos_2psi * g1_values - sin_2psi * g2_values
    g2_transformed = sin_2psi * g1_values + cos_2psi * g2_values

    # Write output (you can choose case here; be consistent downstream)
    with h5py.File(output_hdf5_file, "w") as out_file:
        out_file.create_dataset("ra", data=bg_ra)
        out_file.create_dataset("dec", data=bg_dec)
        out_file.create_dataset("g1", data=g1_transformed)
        out_file.create_dataset("g2", data=g2_transformed)
        if has_weight:
            out_file.create_dataset("weight", data=weights)

    print(f"Transformed background saved to {output_hdf5_file}")





# ------------------------------------------------------------
# INPUT BACKGROUND CATALOG (PARENT DIR)
# ------------------------------------------------------------
base_catalog_dir = os.path.join(parent_dir, "des-data") 

bg_data_path = os.path.join(
    base_catalog_dir, "des-y3-ridges-background-v2.h5"
)

if not os.path.exists(bg_data_path):
    raise FileNotFoundError(
        f"Background catalog not found:\n{bg_data_path}"
    )

# ------------------------------------------------------------
# OUTPUT DIRECTORY (CURRENT DIR)
# ------------------------------------------------------------
noise_dir = os.path.join(current_dir, "noise")
os.makedirs(noise_dir, exist_ok=True)

# ------------------------------------------------------------
# NOISE GENERATION
# ------------------------------------------------------------
num_realizations = 300
base_seed = 123

for i in range(num_realizations):

    seed = base_seed + i
    out_file = os.path.join(
        noise_dir, f"source_catalog_noise_{i:03d}.h5"
    )

    if os.path.exists(out_file):
        print(f"[SKIP] {out_file} already exists")
        continue

    transform_DES_background(
        bg_data_path,
        out_file,
        seed=seed
    )

    print(f"[OK] Created noise catalog → {out_file}")
