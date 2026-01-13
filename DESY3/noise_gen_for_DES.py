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


import numpy as np
import h5py
import re
from ridge_analysis_tools import *



# ------------------------------------------------------------
# INPUT BACKGROUND CATALOG (PARENT DIR)
# ------------------------------------------------------------
base_catalog_dir = os.path.join(parent_dir, "des-data") 

bg_data_path = os.path.join(
    base_catalog_dir, "des-y3-ridges-background-v2_cutzl0.40.h5 "
)

if not os.path.exists(bg_data_path):
    raise FileNotFoundError(
        f"Background catalog not found:\n{bg_data_path}"
    )

# ------------------------------------------------------------
# OUTPUT DIRECTORY (CURRENT DIR)
# ------------------------------------------------------------
noise_dir = os.path.join(current_dir, "noise_zcut")
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
