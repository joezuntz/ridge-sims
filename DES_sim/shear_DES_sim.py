from mpi4py import MPI
import os, sys, glob, re
import numpy as np

# Directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add parent directory to python path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Remain inside this directory for execution
os.chdir(current_dir)

from ridge_analysis_tools import *

COMM = MPI.COMM_WORLD
RANK = COMM.rank

BACKGROUND_TYPE_SIGNAL = "sim"   # RA/DEC/G1/G2/Z_TRUE/weight
BACKGROUND_TYPE_NOISE  = "sim"   # noise catalogs have the same structure

def file_exists(path):
    ok = False
    if RANK == 0:
        ok = os.path.exists(path)
    return COMM.bcast(ok, root=0)

def ensure_dir(path):
    if RANK == 0:
        os.makedirs(path, exist_ok=True)
    COMM.Barrier()



n_random_rotations = 100 # How many noise files to use
# ============================================================
# PATHS
# ============================================================

ridge_file = os.path.join(
    current_dir,
    "DES_sim/run_1/band_0.1/mesh_2/Ridges_final_p15/zero_err_run_1_ridges_p15_contracted.h5"
)
signal_bg  = os.path.join(parent_dir, "lhc_run_sims/run_1/source_catalog_cutzl04.h5")

noise_dir  = os.path.join(current_dir, "DES_sim/noise")
out_dir    = os.path.join(current_dir, "shear/run_1/band_0.1/mesh_2")
ensure_dir(out_dir)

filament_h5   = os.path.join(out_dir, "run_1_p15_filaments.h5")
signal_shear  = os.path.join(out_dir, "run_1_p15_signal_shear.csv")

# random outputs + subtraction output
random_dir = os.path.join(out_dir, "random_rotations")
ensure_dir(random_dir)

subtracted_shear = os.path.join(out_dir, f"run_1_p15_randomsub_N{n_random_rotations:03d}.csv")

# ============================================================
# SIGNAL: build filaments once + compute signal shear 
# ============================================================

if not file_exists(signal_shear) or not file_exists(filament_h5):
    process_ridge_file(
        h5_file=ridge_file,
        BG_data=signal_bg,
        filament_h5=filament_h5,
        shear_csv=signal_shear,
        background_type=BACKGROUND_TYPE_SIGNAL,
        shear_flip_csv=None,
        comm=COMM
    )
COMM.Barrier()

# ============================================================
# NOISE: 
# ============================================================



noise_files_all = sorted(glob.glob(os.path.join(noise_dir, "source_catalog_noise_*.h5")))
if RANK == 0:
    if len(noise_files_all) == 0:
        raise FileNotFoundError(f"No noise catalogs found in: {noise_dir}")
COMM.Barrier()

# Take the first N files 
noise_files = noise_files_all[:n_random_rotations]

if RANK == 0 and len(noise_files) < n_random_rotations:
    print(f"[WARN] Requested n_random_rotations={n_random_rotations}, but only found {len(noise_files)} files.")
COMM.Barrier()

random_csvs = []
for nf in noise_files:
    m = re.search(r"source_catalog_noise_(\d+)\.h5$", os.path.basename(nf))
    rid = int(m.group(1)) if m else -1

    random_csv = os.path.join(random_dir, f"shear_random_{rid:03d}.csv")
    random_csvs.append(random_csv)

    if file_exists(random_csv):
        continue

    # call process_shear_sims directly to reuse filament_h5
    process_shear_sims(
        filament_file=filament_h5,
        bg_data=nf,
        output_shear_file=random_csv,
        k=1, num_bins=20, comm=COMM,
        flip_g1=False, flip_g2=False,
        background_type=BACKGROUND_TYPE_NOISE,
        nside_coverage=32,
        min_distance_arcmin=1.0,
        max_distance_arcmin=60.0
    )

    COMM.Barrier()

# ============================================================
# NOISE SUBTRACTION:
# ============================================================

if RANK == 0:
    # Load signal
    signal = np.loadtxt(signal_shear, delimiter=",", skiprows=1)

    # Load all random profiles that exist
    profiles = []
    for rcsv in random_csvs:
        if not os.path.exists(rcsv):
            raise FileNotFoundError(f"Missing random shear CSV (expected): {rcsv}")
        profiles.append(np.loadtxt(rcsv, delimiter=",", skiprows=1))

    profiles = np.array(profiles)                 # shape: (N, nbins, ncol)
    mean_random = np.mean(profiles, axis=0)       # shape: (nbins, ncol)

    # Consistency checks (binning match)
    if not np.allclose(signal[:, 0], mean_random[:, 0], rtol=0, atol=0):
        raise RuntimeError("Bin_Center mismatch between signal and random mean.")
    if not np.allclose(signal[:, 1], mean_random[:, 1], rtol=0, atol=0):
        print("[WARN] Weighted_Real_Distance differs between signal and random mean (expected in general).")

    g_plus_sub  = signal[:, 2] - mean_random[:, 2]
    g_cross_sub = signal[:, 3] - mean_random[:, 3]

    out = np.column_stack((
        signal[:, 0],   # Bin_Center
        signal[:, 1],   # Weighted_Real_Distance (keep signal's)
        g_plus_sub,
        g_cross_sub,
        signal[:, 4],   # Counts (signal)
        signal[:, 5],   # bin_weight (signal)
    ))

    np.savetxt(
        subtracted_shear,
        out,
        delimiter=",",
        header="Bin_Center,Weighted_Real_Distance,"
               "Weighted_g_plus_subtracted,"
               "Weighted_g_cross_subtracted,"
               "Counts,bin_weight",
        comments="",
    )

    print(f"[rank 0] Saved random-subtracted shear → {subtracted_shear}")

COMM.Barrier()
