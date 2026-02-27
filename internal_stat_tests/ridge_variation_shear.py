from mpi4py import MPI
import os, sys, glob, re
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

# parameters --------------------------------------------------
run_id = 1
BACKGROUND_TYPE_SIGNAL = "sim"

# Input Background file 
signal_bg = os.path.join(parent_dir, "lhc_run_sims_zero_err_10", f"run_{run_id}", "source_catalog_cutzl04.h5")

# Find contracted ridge files
ridges_dir = os.path.join(current_dir, "ridge_variation", "Ridges_final_p15")
ridge_files = sorted(glob.glob(os.path.join(ridges_dir, "*_contracted.h5")))

# Output directory
out_dir = os.path.join(current_dir, "ridge_variation_shear")
if RANK == 0:
    os.makedirs(out_dir, exist_ok=True)
COMM.Barrier()

for ridge_file in ridge_files:
    # extract tag "00".."49" from "..._p15_XX_contracted.h5"
    m = re.search(r"_p15_(\d+)_contracted\.h5$", os.path.basename(ridge_file))
    tag = m.group(1) if m else "xx"

    filament_h5  = os.path.join(out_dir, f"filaments_p15_{tag}.h5")
    signal_shear = os.path.join(out_dir, f"signal_shear_p15_{tag}.csv")

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