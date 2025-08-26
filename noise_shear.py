import os
import h5py
from ridge_analysis_tools import *
import matplotlib.pyplot as plt

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

# Directories
base_sim_dir = "lhc_run_sims"
run_id = 1
noise_shear_dir = "example_zl04_mesh5e5/noise/shear"
filament_dir = "example_zl04_mesh5e5/filaments"
os.makedirs(filament_dir, exist_ok=True)

# Final density 
final_percentiles = [15]

# Loop over final_percentiles (here just 15)
for fp in final_percentiles:
    if comm is None or comm.rank == 0:
        print(f"[rank 0] Processing filaments for final_percentile={fp}")
        h5_file = f"example_zl04_mesh5e5/Ridges_final_p{fp:02d}/ridges_p{fp:02d}.h5"
        with h5py.File(h5_file, 'r') as f:
            Ridges = f["ridges"][:]

        # Build MST + segment filaments
        mst = build_mst(Ridges)
        branch_points = detect_branch_points(mst)
        filament_segments = split_mst_at_branches(mst, branch_points)
        filament_labels = segment_filaments_with_dbscan(Ridges, filament_segments)

        # Save filament segmentation
        filament_h5 = os.path.join(filament_dir, f"filaments_p{fp:02d}.h5")
        save_filaments_to_hdf5(Ridges, filament_labels, filament_h5)

    if comm is not None:
        comm.Barrier()

    filament_h5 = os.path.join(filament_dir, f"filaments_p{fp:02d}.h5")

    # --- Loop over 30 noise realizations ---
    num_realizations = 30
    for i in range(num_realizations):
        noise_file = os.path.join(noise_shear_dir, f"source_catalog_noise_{i:02d}.h5")
        shear_noise_csv = os.path.join(
            noise_shear_dir, f"shear_noise_p{fp:02d}_r{i:02d}.csv"
        )

        if comm is None or comm.rank == 0:
            print(f"[rank 0] Processing noise realization {i} for p={fp}")
        
        process_shear_sims(filament_h5, noise_file, output_shear_file=shear_noise_csv, background_type ='sim')
        process_shear_sims(filament_h5, noise_file, output_shear_file=shear_noise_csv.replace(".csv", "_flipG1.csv"), flip_g1=True, background_type='sim')
    if comm is not None:
        comm.Barrier()
