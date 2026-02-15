import os, sys




current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
os.chdir(current_dir)

import glob
from mpi4py import MPI
from ridge_analysis_tools import *
comm = MPI.COMM_WORLD
rank = comm.rank





#def find_contracted_files(root):
#    out = []
#    for r, _, files in os.walk(root):
#        for f in files:
#            if f.endswith("_contracted.h5"):
#                out.append(os.path.join(r, f))
#    return sorted(out)



def main():

    ridge_root = "shear_stat_test"

    ridges_dir = os.path.join(
        ridge_root,
        "run_1_mesh_2_band_0.10",
        "Ridges_final_p15",
    )

    h5_file = glob.glob(os.path.join(ridges_dir, "*_contracted.h5"))[0]

    BG_data = os.path.join(
        parent_dir, "lhc_run_sims_zero_err_10", "run_1", "source_catalog_cutzl04.h5"
    )

    out_root = os.path.join(ridge_root, "shear_repeats")
    if rank == 0:
        os.makedirs(out_root, exist_ok=True)

    for shear in range(50):
        process_ridge_file(
            h5_file=h5_file,
            BG_data=BG_data,
            filament_h5=os.path.join(out_root, f"filaments_{shear:02d}.h5"),
            shear_csv=os.path.join(out_root, f"shear_{shear:02d}.csv"),
            background_type="sim",
            shear_flip_csv=None,
            comm=comm,
        )

    comm.Barrier()

if __name__ == "__main__":
    main()
