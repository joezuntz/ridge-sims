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


import numpy as np
import h5py
from ridge_analysis_tools import * 


# After execution the folder should look something like:
"""
lhc_cosmo_sims_zero_err/
├── S8/
│   ├── run_1/
│   │   ├── source_catalog_0.npy
│   │   └── source_catalog_cutzl04.h5
│   ├── run_2/
│   │   ├── source_catalog_0.npy
│   │   └── source_catalog_cutzl04.h5
│   └── ...
├── Om_fixed/
│   ├── run_1/
...
"""



############################################################
# === Apply conversion to all cosmology runs
############################################################


if __name__ == "__main__":
    base_sim_root = os.path.join(parent_dir, "lhc_cosmo_sims_zero_err")
    convert_all_backgrounds(base_sim_root)