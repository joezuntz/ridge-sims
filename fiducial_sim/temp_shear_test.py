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


n_random_rotations = 100

BACKGROUND_TYPE_SIGNAL = "sim"
BACKGROUND_TYPE_NOISE  = "sim"



# Input ridge file
ridge_file = "density_threshold_test/run_1_mesh_2_band_0.1/Ridges_final_p40/shape_error_run_1_ridges_p40_contracted.h5"


# Input Background file
signal_bg = os.path.join(parent_dir, "lhc_DES_fiducial_sim", f"run_{run_id}", "source_catalog_cutzl0.40.h5")
#noise_dir = os.path.join(parent_dir, "DES_sim/DES_sim/noise")  # same noise files as previous sims

# Output shear directory 
out_dir = "density_threshold_test"
#random_dir = os.path.join(out_dir, "random_rotations")

if RANK == 0:
    os.makedirs(out_dir, exist_ok=True)
    #os.makedirs(random_dir, exist_ok=True)
COMM.Barrier()

filament_h5  = os.path.join(out_dir, f"filaments_p40.h5")
signal_shear = os.path.join(out_dir, f"signal_shear_p40.csv")

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
        skip_end_points=False, 
        min_filament_points=0,
        shear_flip_csv=None,    
        comm=COMM
    )
COMM.Barrier()