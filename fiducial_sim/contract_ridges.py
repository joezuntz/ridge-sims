import os, sys
import numpy as np
from mpi4py import MPI

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
os.chdir(current_dir)

import dredge_scms
from ridge_analysis_tools import *

COMM = MPI.COMM_WORLD
RANK = COMM.rank


def main():
    
    
    base_label = "shape_err"
    run_id = 1
    N = 2
    bandwidth = 0.1
    
    # Output base
    output_base = "DES_fiducial_sim"
    home_dir = os.path.join(
        output_base,
        f"band_{bandwidth:.1f}_mesh_{N}",
    )

    ridges_dir = os.path.join(home_dir, "Ridges_final_p15")
    plots_dir  = os.path.join(home_dir, "plots_by_final_percentile")
    
    # ridge_file for contraction
    ridge_file = os.path.join(
        ridges_dir,
        f"{base_label}_run_{run_id}_ridges_p15.h5"
    )

    # Contraction params

    radius_arcmin = 4.0
    min_coverage = 0.9
    nside = 512

    mask_filename = os.path.join(parent_dir, "des-data", "desy3_gold_mask.npy")
    mask = np.load(mask_filename) if RANK == 0 else None
    mask = COMM.bcast(mask, root=0)

    

    COMM.barrier() 

    # Contraction
    if RANK == 0:
        process_ridge_file_local(
            ridge_file,
            mask,
            nside,
            radius_arcmin,
            min_coverage,
            ridges_dir,
            plots_dir
        )
        
if __name__ == "__main__":
    main()

