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
    start_time = time.time()   # start timer
    
    base_sim_dir = "lhc_run_sims_50"
    num_runs = 3
    num_noises = 300       # number of noise realizations per run
    final_percentile = 15  # fixed

    for run_id in range(1, num_runs + 1):
        print(f"\n=== Processing RUN {run_id} (percentile {final_percentile}) ===")

        # Background file for this run
        BG_data = os.path.join(base_sim_dir, f"run_{run_id}", "source_catalog_cutzl04.h5")

        # Match directory structure
        output_base_dir = f"example30_band0.4/8test/run_{run_id}"
        noise_dir = f"example30_band0.4/8test/noise_data"
        filament_dir = os.path.join(output_base_dir, "filaments_filtered")
        os.makedirs(filament_dir, exist_ok=True)

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
        shear_csv = os.path.join(filament_dir, f"3runs/shear_p{final_percentile:02d}.csv")
        shear_noise_csv = os.path.join(filament_dir, f"3runs/shear_noise_p{final_percentile:02d}.csv")
        os.makedirs(os.path.dirname(shear_csv), exist_ok=True)

        # Compute shear (signal only)
        process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv, background_type="sim")

        # === Compute noise shear across all realizations ===
        print(f"Run {run_id}: Processing {num_noises} noise realizations...")
        all_noise_shears = []

        for i in range(num_noises):
            noise_file = os.path.join(noise_dir, f"noise_r{run_id:02d}_n{i:03d}.h5")
            tmp_csv = shear_noise_csv.replace(".csv", f"_n{i:03d}.csv")
            
            process_shear_sims(filament_h5, noise_file, output_shear_file=tmp_csv, background_type="sim")
            noise_data = np.loadtxt(tmp_csv, delimiter=",", skiprows=1)
            all_noise_shears.append(noise_data)

        # Stack and average noise across all realizations
        all_noise_shears = np.array(all_noise_shears)  # shape = (num_noises, bins, cols)
        mean_noise = np.mean(all_noise_shears, axis=0)

        np.savetxt(
            shear_noise_csv,
            mean_noise,
            delimiter=",",
            header="Bin_Center,Weighted_Real_Distance,Weighted_g_plus,Weighted_g_cross,Counts,bin_weight",
            comments=""
        )
        print(f"Run {run_id}: Averaged noise shear saved → {shear_noise_csv}")

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
                filament_dir,"3runs", f"shear_p{final_percentile:02d}_shear-randomshear.csv"
            )
            os.makedirs(os.path.dirname(subtracted_output_file), exist_ok=True)
            np.savetxt(
                subtracted_output_file,
                subtracted_data,
                delimiter=",",
                header="Bin_Center,Weighted_Real_Distance,Weighted_g_plus_subtracted,Weighted_g_cross_subtracted,Counts,bin_weight",
                comments=""
            )

            print(f"Run {run_id}: Subtracted shear profiles saved to {subtracted_output_file}")
    # End timing
    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n=== Script finished in {minutes} min {seconds} sec ===")


if __name__ == "__main__":
    main()
