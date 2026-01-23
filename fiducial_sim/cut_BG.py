import os
import sys


# PATH SETUP
# Directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# One level up 
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
# Make parent importable 
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import h5py
from ridge_analysis_tools import * 



# MAIN


if __name__ == "__main__":

    sim_root_name = "lhc_DES_fiducial_sim/run_1"
    base_root = os.path.join(parent_dir, sim_root_name)

    # Apply background cut / conversion
    convert_all_backgrounds(base_root, z_cut=0.4)

    print("\n[INFO] Done.")
