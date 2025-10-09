"""
Runs shear computation for all runs and bands, alternating between
normal and zero-error directories for each band.

Folder structure created inside each band:
band_X/shear_calc/
    ├── raw_ridges_shear/
    │   ├── filament_segments/
    │   └── noise_shear/
    └── shrinked_ridges_shear/
        ├── filament_segments/
        └── noise_shear/
"""

import os
import h5py
import numpy as np
import time
from ridge_analysis_tools import *
from mpi4py import MPI

comm = MPI.COMM_WORLD if MPI.Is_initialized() else None

# === CONFIGURATION ===
bands = ["band_0.1"]
run_ids = [1]
final_percentiles = [15]
base_root = "simulation_ridges_comparative_analysis"
noise_dir = "example_zl04_mesh5e5/noise"

# Collect all noise realizations
noise_files = sorted(
    [f for f in os.listdir(noise_dir)
     if f.startswith("source_catalog_noise_") and f.endswith(".h5")],
    key=lambda x: int(x.split("_")[-1].replace(".h5", ""))
)

# ======================================================
# === Compute shear for a given ridge file ===
# ======================================================
def compute_shear_for_ridges(ridges_h5, bg_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filament_dir = os.path.join(output_dir, "filament_segments")
    noise_shear_dir = os.path.join(output_dir, "noise_shear")
    os.makedirs(filament_dir, exist_ok=True)
    os.makedirs(noise_shear_dir, exist_ok=True)

    fp = 15  # single percentile
    with h5py.File(ridges_h5, "r") as f:
        ridges = f["ridges"][:]

    # --- Build filaments ---
    mst = build_mst(ridges)
    branch_points = detect_branch_points(mst)
    filament_segments = split_mst_at_branches(mst, branch_points)
    filament_labels = segment_filaments_with_dbscan(ridges, filament_segments)
    filament_h5 = os.path.join(filament_dir, f"filaments_p{fp:02d}.h5")
    save_filaments_to_hdf5(ridges, filament_labels, filament_h5)

    # === File paths for signal shear ===
    shear_csv = os.path.join(filament_dir, f"shear_p{fp:02d}.csv")
    shear_flip_csv = os.path.join(filament_dir, f"shear_p{fp:02d}_flipG1.csv")

    # --- Run with signal ---
    process_shear_sims(filament_h5, bg_data,
                       output_shear_file=shear_csv,
                       background_type='sim', plot_output_dir=filament_dir)
    process_shear_sims(filament_h5, bg_data,
                       output_shear_file=shear_flip_csv,
                       flip_g1=True, background_type='sim')

    # === Loop over noise realizations ===
    all_noise_profiles, all_noise_flip_profiles = [], []
    for nf in noise_files:
        realization_id = nf.split("_")[-1].replace(".h5", "")
        noise_file = os.path.join(noise_dir, nf)

        shear_noise_csv_i = os.path.join(
            noise_shear_dir, f"shear_noise_p{fp:02d}_{realization_id}.csv"
        )
        shear_noise_flip_csv_i = os.path.join(
            noise_shear_dir, f"shear_noise_p{fp:02d}_{realization_id}_flipG1.csv"
        )

        process_shear_sims(filament_h5, noise_file,
                           output_shear_file=shear_noise_csv_i,
                           background_type='sim')
        process_shear_sims(filament_h5, noise_file,
                           output_shear_file=shear_noise_flip_csv_i,
                           flip_g1=True, background_type='sim')

        all_noise_profiles.append(
            np.loadtxt(shear_noise_csv_i, delimiter=",", skiprows=1))
        all_noise_flip_profiles.append(
            np.loadtxt(shear_noise_flip_csv_i, delimiter=",", skiprows=1))

    # === Compute mean noise and subtract ===
    all_noise_profiles = np.array(all_noise_profiles)
    all_noise_flip_profiles = np.array(all_noise_flip_profiles)
    mean_noise = np.mean(all_noise_profiles, axis=0)
    mean_noise_flip = np.mean(all_noise_flip_profiles, axis=0)

    shear_data = np.loadtxt(shear_csv, delimiter=",", skiprows=1)
    shear_data_flip = np.loadtxt(shear_flip_csv, delimiter=",", skiprows=1)

    g_plus_subtracted = shear_data[:, 2] - mean_noise[:, 2]
    g_cross_subtracted = shear_data[:, 3] - mean_noise[:, 3]
    g_plus_subtracted_flip = shear_data_flip[:, 2] - mean_noise_flip[:, 2]
    g_cross_subtracted_flip = shear_data_flip[:, 3] - mean_noise_flip[:, 3]

    subtracted_data = np.column_stack((
        shear_data[:, 0], shear_data[:, 1],
        g_plus_subtracted, g_cross_subtracted,
        shear_data[:, 4], shear_data[:, 5]
    ))
    np.savetxt(
        os.path.join(filament_dir, f"shear_p{fp:02d}_shear-randomshear.csv"),
        subtracted_data, delimiter=",",
        header="Bin_Center,Weighted_Real_Distance,Weighted_g_plus_subtracted,Weighted_g_cross_subtracted,Counts,bin_weight",
        comments=""
    )

    subtracted_data_flip = np.column_stack((
        shear_data_flip[:, 0], shear_data_flip[:, 1],
        g_plus_subtracted_flip, g_cross_subtracted_flip,
        shear_data_flip[:, 4], shear_data_flip[:, 5]
    ))
    np.savetxt(
        os.path.join(filament_dir, f"shear_p{fp:02d}_flipG1_shear-randomshear.csv"),
        subtracted_data_flip, delimiter=",",
        header="Bin_Center,Weighted_Real_Distance,Weighted_g_plus_subtracted,Weighted_g_cross_subtracted,Counts,bin_weight",
        comments=""
    )

    if comm is None or comm.rank == 0:
        print(f"Finished shear computation for {ridges_h5}")


# ================================================
# === MAIN LOOP ===
# ================================================
def main():
    for band in bands:
        for run_id in run_ids:
            for variant in ["normal", "zero_err"]:
                if comm is None or comm.rank == 0:
                    print(f"\n--- Running {band}/{variant}/run_{run_id} ---")

                base_dir = os.path.join(base_root, variant, band)
                shear_calc_dir = os.path.join(base_dir, "shear_calc")
                os.makedirs(shear_calc_dir, exist_ok=True)

                for ridge_type in ["raw_ridges", "shrinked_ridges"]:
                    if ridge_type == "raw_ridges":
                        ridge_subdir = "Ridges_final_p15"
                        ridge_filename = f"{variant}_run_{run_id}_ridges_p15.h5"
                    else:
                        ridge_subdir = "Shrinked_Ridges_final_p15"
                        ridge_filename = f"{variant}_run_{run_id}_ridges_p15_shrinked.h5"

                    ridge_h5 = os.path.join(base_dir, ridge_subdir, ridge_filename)

                    if not os.path.exists(ridge_h5):
                        if comm is None or comm.rank == 0:
                            print(f"[WARN] Ridge file not found: {ridge_h5}, skipping.")
                        continue

                    bg_data = os.path.join(base_dir, "source_catalog_cutzl04.h5")
                    output_dir = os.path.join(shear_calc_dir, f"{ridge_type}_shear")
                    compute_shear_for_ridges(ridge_h5, bg_data, output_dir)


if __name__ == "__main__":
    start = time.time()
    main()
    if comm is None or comm.rank == 0:
        print(f"\nTotal time: {time.time() - start:.1f} sec")
