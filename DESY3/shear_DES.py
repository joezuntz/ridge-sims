import os
import sys
import numpy as np
from mpi4py import MPI

# ==============================================================
# PATH SETUP
# ==============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

os.chdir(current_dir)

from ridge_analysis_tools import *

COMM = MPI.COMM_WORLD
RANK = COMM.rank





# ------------------------------------------------------------
# INPUTS 
# ------------------------------------------------------------
# contracted ridge file 
h5_file = os.path.join(current_dir, "DES_ridge_analysis/Ridges_analysis/DESY3_ridges_p15__mesh2_band0.10_contracted_update.h5")   #Update  

# Output directories
filament_dir = os.path.join(current_dir, "filaments_update_noshift")                         # Update
shear_dir    = os.path.join(current_dir, "shear_update_noshift")                             # Update
os.makedirs(filament_dir, exist_ok=True)
os.makedirs(shear_dir, exist_ok=True)

# cut background
base_catalog_dir = os.path.join(parent_dir, "des-data")
bg_file = os.path.join(base_catalog_dir, "des-y3-ridges-background-v2_cutzl0.40.h5")

# Noise catalogs directory
noise_dir = os.path.join(current_dir, "noise_zcut")

# Parameters
final_percentiles = [15]
n_random_rotations = 100

# ------------------------------------------------------------
# safety checks
# ------------------------------------------------------------
if RANK == 0:
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"Missing ridge file: {h5_file}")
    if not os.path.exists(bg_file):
        raise FileNotFoundError(f"Missing background file: {bg_file}")
    if not os.path.isdir(noise_dir):
        raise FileNotFoundError(f"Missing noise directory: {noise_dir}")

COMM.Barrier()

# ============================================================
# LOOP OVER FILAMENT SETS
# ============================================================
for fp in final_percentiles:

    filament_h5 = os.path.join(filament_dir, f"filaments_p{fp:02d}.h5")
    shear_csv   = os.path.join(shear_dir,    f"shear_p{fp:02d}.csv")

    # --------------------------------------------------------
    # SIGNAL 
    # --------------------------------------------------------
    process_ridge_file(
        h5_file=h5_file,
        BG_data=bg_file,
        filament_h5=filament_h5,
        shear_csv=shear_csv,
        background_type="DES_noshift",
        shear_flip_csv=None,
        comm=COMM,
    )

    COMM.Barrier()

    # flip signs 
    
    shear_flip_csv = os.path.join(shear_dir, f"shear_p{fp:02d}_flipg1.csv")  

    process_ridge_file(
        h5_file=h5_file,
        BG_data=bg_file,
        filament_h5=filament_h5,
        shear_csv=shear_csv,                 
        background_type="DES_noshift",
        shear_flip_csv=shear_flip_csv,       
        comm=COMM,
    )
    
    COMM.Barrier()

    COMM.Barrier()

    # --------------------------------------------------------
    # NOISE 
    # --------------------------------------------------------
#    all_random_profiles = []

#    for i in range(n_random_rotations):
#        noise_bg = os.path.join(noise_dir, f"source_catalog_noise_{i:03d}.h5")
#        random_csv = os.path.join(shear_dir, f"shear_random_p{fp:02d}_{i:03d}.csv")

#        if not os.path.exists(noise_bg):
#            raise FileNotFoundError(f"Missing noise catalog: {noise_bg}")

#        process_ridge_file(
#            h5_file=h5_file,
#            BG_data=noise_bg,
#            filament_h5=filament_h5,
#            shear_csv=random_csv,
#            background_type="noise_noshift",
#            shear_flip_csv=None,
#            comm=COMM,
#        )

#        COMM.Barrier()

#        if RANK == 0:
#            all_random_profiles.append(np.loadtxt(random_csv, delimiter=",", skiprows=1))

#    # --------------------------------------------------------
#    # SUBTRACTION (rank 0 only)
#    # --------------------------------------------------------
#    if RANK == 0:
#        all_random_profiles = np.array(all_random_profiles)
#        mean_random = np.mean(all_random_profiles, axis=0)

#        shear_data = np.loadtxt(shear_csv, delimiter=",", skiprows=1)

#        g_plus_sub  = shear_data[:, 2] - mean_random[:, 2]
#        g_cross_sub = shear_data[:, 3] - mean_random[:, 3]

#        out = np.column_stack((
#            shear_data[:, 0],
#            shear_data[:, 1],
#            g_plus_sub,
#            g_cross_sub,
#            shear_data[:, 4],
#            shear_data[:, 5],
#        ))

#        out_file = os.path.join(shear_dir, f"shear_p{fp:02d}_randomsub.csv")

#        np.savetxt(
#            out_file,
#            out,
#            delimiter=",",
#            header="Bin_Center,Weighted_Real_Distance,"
#                   "Weighted_g_plus_subtracted,"
#                   "Weighted_g_cross_subtracted,"
#                   "Counts,bin_weight",
#            comments="",
#        )

#        print(f"[rank 0] Saved random-subtracted shear → {out_file}")

#    COMM.Barrier()
