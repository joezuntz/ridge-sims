import os


import os, sys

# Directory of this script (cosmo_sims)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up 
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# find modules in the parent directory
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# we stay inside cosmo_sims for outputs
os.chdir(current_dir)



import h5py
import numpy as np
import matplotlib.pyplot as plt

# ---- FILES ----
bg_file = os.path.abspath(
    os.path.join(parent_dir, "lhc_cosmo_sims2_zero_err/S8/run_3/source_catalog_0.npy")
)

ridge_file = os.path.abspath(
    os.path.join(current_dir, "Cosmo_sim2_ridges/S8/run_3/band_0.1/Ridges_final_p15/S8_run_3_ridges_p15.h5")
)
output_path = "Cosmo_sim2_ridges/overlay_test_S8_run3.png"

# ---- LOAD BACKGROUND ----
with h5py.File(bg_file, "r") as f:
    print("BG keys:", list(f.keys()))
    ra_bg  = f["ra"][:]
    dec_bg = f["dec"][:]

# ---- LOAD RIDGES ----
with h5py.File(ridge_file, "r") as f:
    ridges = f["ridges"][:]

dec_ridge = ridges[:, 0]
ra_ridge  = ridges[:, 1]

# ---- PRINT RANGES (diagnostic) ----
print("Background RA range:", ra_bg.min(), ra_bg.max())
print("Background Dec range:", dec_bg.min(), dec_bg.max())
print("Ridge RA range:", ra_ridge.min(), ra_ridge.max())
print("Ridge Dec range:", dec_ridge.min(), dec_ridge.max())

# ---- PLOT ----
plt.figure(figsize=(8,6))

plt.scatter(ra_bg, dec_bg, s=1, alpha=0.05, label="Background")
plt.scatter(ra_ridge, dec_ridge, s=1, alpha=0.6, label="Ridges")

plt.xlabel("RA")
plt.ylabel("Dec")
plt.legend()
plt.tight_layout()

os.makedirs("Cosmo_sim2_ridges", exist_ok=True)
plt.savefig(output_path, dpi=200)
plt.close()

print(f"Saved overlay plot → {output_path}")