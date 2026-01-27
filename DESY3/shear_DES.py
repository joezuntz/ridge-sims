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


# Temporary helper 

def process_ridge_file_DESY3(h5_file, BG_data, filament_h5, shear_csv = None, background_type, shear_flip_csv = None, comm=None):
    """
    Compute MST → filaments → shear from a contracted ridge file.
    All paths are passed explicitly to keep the function file-agnostic.
    """
    if comm is None or comm.rank == 0:
        print(f"[rank 0] Processing {h5_file}")

        with h5py.File(h5_file, 'r') as f:
            Ridges = f["ridges"][:]

        mst = build_mst(Ridges)
        branch_points = detect_branch_points(mst)
        filament_segments = split_mst_at_branches(mst, branch_points)
        filament_labels = segment_filaments_with_dbscan(Ridges, filament_segments)

        save_filaments_to_hdf5(Ridges, filament_labels, filament_h5)
        print(f"[save] Filaments → {filament_h5}")

    if comm is not None:
        comm.Barrier()

    # --- Shear calculations ---
    if shear_csv is not None:
        process_shear_sims(
            filament_h5, BG_data, output_shear_file=shear_csv,
            k=1, num_bins=20, comm=comm,
            flip_g1=False, flip_g2=False, background_type=background_type,
            nside_coverage=32, min_distance_arcmin=1.0, max_distance_arcmin=60.0
        )

    if shear_flip_csv is not None:
        process_shear_sims(
            filament_h5, BG_data, output_shear_file=shear_flip_csv,
            k=1, num_bins=20, comm=comm,
            flip_g1=False, flip_g2=True, background_type=background_type,
            nside_coverage=32, min_distance_arcmin=1.0, max_distance_arcmin=60.0
        )


R_CAL = 0.6805452481
def apply_R_calibration_inplace(csv_path, R=R_CAL):
    """
    Divide Weighted_g_plus and Weighted_g_cross by R.
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    data[:, 2] /= R  # Weighted_g_plus
    data[:, 3] /= R  # Weighted_g_cross

    header = "Bin_Center,Weighted_Real_Distance,Weighted_g_plus,Weighted_g_cross,Counts,bin_weight"
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

# ------------------------------------------------------------
# INPUTS 
# ------------------------------------------------------------
# contracted ridge file 
h5_file = os.path.join(current_dir, "DES_ridge_analysis/Ridges_analysis/DESY3_ridges_p15__mesh2_band0.10_contracted_update.h5")   #Update  

# Output directories
filament_dir = os.path.join(current_dir, "filaments_flipg2")                         # Update
shear_dir    = os.path.join(current_dir, "shear_flipg2")                             # Update
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

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------
    shear_flip_csv = os.path.join(shear_dir, f"shear_p{fp:02d}_flipg2.csv")

    # Compute filaments only if missing
    if (RANK == 0) and os.path.exists(filament_h5):
        print(f"[rank 0] Reusing filaments -> {filament_h5}")
    else:
        process_ridge_file_DESY3(
            h5_file=h5_file,
            BG_data=bg_file,
            filament_h5=filament_h5,
            shear_csv=None,
            background_type="DES_noshift",
            shear_flip_csv=shear_flip_csv,
            comm=COMM,
        )

    COMM.Barrier()

    if RANK == 0:
        apply_R_calibration_inplace(shear_flip_csv)
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

#        process_shear_sims(
#            filament_h5, noise_bg, output_shear_file=random_csv,
#            k=1, num_bins=20, comm=COMM,
#            flip_g1=False, flip_g2=True,
#            background_type="noise_noshift",
#            nside_coverage=32, min_distance_arcmin=1.0, max_distance_arcmin=60.0
#        )

#        COMM.Barrier()
#        if RANK == 0:
#            apply_R_calibration_inplace(random_csv)
#            all_random_profiles.append(np.loadtxt(random_csv, delimiter=",", skiprows=1))
#        COMM.Barrier()