import os
import pandas as pd
import numpy as np
import h5py
import time
from ridge_analysis_tools import *
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

def main():
    base_sim_dir = "lhc_run_sims_50"
    num_runs = 4
    final_percentile = 15  # fixed

    for run_id in range(1, num_runs + 1):
        print(f"\n=== Processing RUN {run_id} (percentile {final_percentile}) ===")

        # Background file for this run
        BG_data = os.path.join(base_sim_dir, f"run_{run_id}", "source_catalog_cutzl04.h5")

        # Match directory structure !
        output_base_dir = f"example30_band0.4/8test/run_{run_id}"
        filament_dir = os.path.join(output_base_dir, "filaments_filtered")
        os.makedirs(filament_dir, exist_ok=True)

        # Noise directory (per run)
        noise_dir = os.path.join(output_base_dir, "noise")
        os.makedirs(noise_dir, exist_ok=True)
        noise = os.path.join(noise_dir, "source_catalog_noise.h5")

        # Ridges file path
        h5_file = os.path.join(output_base_dir, f"ridges_filtered/ridges_p{final_percentile:02d}_filtered.h5")

        if comm is None or comm.rank == 0:
            print(f"[rank 0] Run {run_id}: Building filaments")
            with h5py.File(h5_file, "r") as f:
                Ridges = f["ridges"][:]

            mst = build_mst(Ridges)
            branch_points = detect_branch_points(mst)
            filament_segments = split_mst_at_branches(mst, branch_points)
            filament_labels = segment_filaments_with_dbscan(Ridges, filament_segments)

            filament_h5 = os.path.join(filament_dir, f"filaments_p{final_percentile:02d}.h5")
            save_filaments_to_hdf5(Ridges, filament_labels, filament_h5)

        if comm is not None:
            comm.Barrier()

        # Paths for shear outputs
        filament_h5 = os.path.join(filament_dir, f"filaments_p{final_percentile:02d}.h5")
        shear_csv = os.path.join(filament_dir, f"4runs/shear_p{final_percentile:02d}.csv")
        shear_noise_csv = os.path.join(noise_dir, f"4runs/shear_noise_p{final_percentile:02d}.csv")
        os.makedirs(shear_csv, exist_ok=True)
        os.makedirs(shear_noise_csv, exist_ok=True)
        # Compute shear (signal + noise)
        process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv, background_type="sim")
        process_shear_sims(filament_h5, noise, output_shear_file=shear_noise_csv, background_type="sim")

        # === Subtract noise ===
        if comm is None or comm.rank == 0:
            print(f"Run {run_id}: Subtracting noise from shear profiles")

            shear_data = np.loadtxt(shear_csv, delimiter=",", skiprows=1)
            noise_data = np.loadtxt(shear_noise_csv, delimiter=",", skiprows=1)

            g_plus_signal = shear_data[:, 2]
            g_cross_signal = shear_data[:, 3]
            g_plus_noise = noise_data[:, 2]
            g_cross_noise = noise_data[:, 3]

            g_plus_subtracted = g_plus_signal - g_plus_noise
            g_cross_subtracted = g_cross_signal - g_cross_noise

            subtracted_data = np.column_stack((
                shear_data[:, 0],  # Bin_Center
                shear_data[:, 1],  # Weighted_Real_Distance
                g_plus_subtracted,
                g_cross_subtracted,
                shear_data[:, 4],  # Counts
                shear_data[:, 5]   # bin_weight
            ))

            subtracted_output_file = os.path.join(
                filament_dir, f"shear_p{final_percentile:02d}_shear-randomshear.csv"
            )

            np.savetxt(
                subtracted_output_file,
                subtracted_data,
                delimiter=",",
                header="Bin_Center,Weighted_Real_Distance,Weighted_g_plus_subtracted,Weighted_g_cross_subtracted,Counts,bin_weight",
                comments=""
            )

            print(f"Run {run_id}: Subtracted shear profiles saved to {subtracted_output_file}")


if __name__ == "__main__":
    main()
