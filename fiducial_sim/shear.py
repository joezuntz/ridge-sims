from mpi4py import MPI
import os, sys, glob
import numpy as np

# Path setup --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
os.chdir(current_dir)

from ridge_analysis_tools import *

COMM = MPI.COMM_WORLD
RANK = COMM.rank


# parameters

base_label = "shape_err"
run_id = 1
N = 2
bandwidth = 0.1
p_final = 15

n_random_rotations = 100

BACKGROUND_TYPE_SIGNAL = "sim"
BACKGROUND_TYPE_NOISE  = "sim"

# folders


ridge_home = os.path.join("DES_fiducial_sim", f"band_{bandwidth:.1f}_mesh_{N}")

ridges_dir = os.path.join(ridge_home, f"Ridges_final_p{p_final}")

# Input ridge file
ridge_file = os.path.join(
    ridges_dir,
    f"{base_label}_run_{run_id}_ridges_p{p_final}_contracted.h5"
)

# Input Background file
signal_bg = os.path.join(parent_dir, "lhc_DES_fiducial_sim", f"run_{run_id}", "source_catalog_cutzl0.40.h5")
noise_dir = os.path.join(parent_dir, "DES_sim/DES_sim/noise")  # same noise files as previous sims

# Output shear directory 
#out_dir = os.path.join("shear", ridge_home, f"run_{run_id}_p{p_final}")
out_dir = "shear_no_endpoint"
random_dir = os.path.join(out_dir, "random_rotations")

if RANK == 0:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(random_dir, exist_ok=True)
COMM.Barrier()

filament_h5  = os.path.join(out_dir, f"filaments.h5")
signal_shear = os.path.join(out_dir, f"signal_shear.csv")

# Build filaments + Signal shear 

need_signal = True
if RANK == 0:
    need_signal = (not os.path.exists(filament_h5)) or (not os.path.exists(signal_shear))
need_signal = COMM.bcast(need_signal, root=0)

if need_signal:
    process_ridge_file(
        h5_file=ridge_file,
        BG_data=signal_bg,
        filament_h5=filament_h5,
        shear_csv=signal_shear,
        background_type=BACKGROUND_TYPE_SIGNAL,
        skip_end_points=True, 
        min_filament_points=3,
        shear_flip_csv=None,    
        comm=COMM
    )
COMM.Barrier()

# Noise shear 


#noise_files = sorted(glob.glob(os.path.join(noise_dir, "source_catalog_noise_*.h5")))


#noise_files = noise_files[:n_random_rotations]

#for i, nf in enumerate(noise_files):
#    random_csv = os.path.join(random_dir, f"shear_random_{i:03d}.csv")

#    exists = True
#    if RANK == 0:
#        exists = os.path.exists(random_csv)
#    exists = COMM.bcast(exists, root=0)
#    if exists:
#        continue

#    process_shear_sims(
#        filament_file=filament_h5,
#        bg_data=nf,
#        output_shear_file=random_csv,
#        k=1, num_bins=20, comm=COMM,
#        flip_g1=False, flip_g2=False,
#        background_type=BACKGROUND_TYPE_NOISE,
#        nside_coverage=32,
#        min_distance_arcmin=1.0,
#        max_distance_arcmin=60.0
#    )
#    COMM.Barrier()
