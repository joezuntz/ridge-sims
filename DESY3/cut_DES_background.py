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


# ==============================================================
# User parameters
# ==============================================================
BG_FILE = os.path.join(parent_dir,"des-data/des-y3-ridges-background-v2.h5")
Z_CUT   = 0.4

# ==============================================================
# Run
# ==============================================================
if __name__ == "__main__":

    if not os.path.exists(BG_FILE):
        raise FileNotFoundError(f"Input file not found: {BG_FILE}")

    out_file = convert_DES_background_with_zcut(
        BG_FILE,
        z_cut=Z_CUT,
        comm=None,   
    )

    print(f"[OK] Written: {out_file}")