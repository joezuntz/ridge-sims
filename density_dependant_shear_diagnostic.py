import os
import pandas as pd
import numpy as np
import h5py
import time
from ridge_analysis.tools import *
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt  

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

# Load background data for shear
base_sim_dir = "lhc_run_sims"
run_id = 1
BG_data = os.path.join(base_sim_dir, f"run_{run_id}", "source_catalog_0.npy")

def process_shear_sims(filament_file, bg_data, output_shear_file, k=1, num_bins=20, comm=comm,
                       flip_g1=False, flip_g2=False):  # Added flip sign flags
    start_time = time.time()

    # Load filament data
    with h5py.File(filament_file, "r") as hdf:
        dataset = hdf["data"]
        ra_values = dataset["RA"][:]
        dec_values = dataset["DEC"][:]
        labels = dataset["Filament_Label"][:]
        rows = dataset["RA"].size

    unique_labels = np.unique(labels)

    with h5py.File(bg_data, "r") as file:
        if comm is None:
            s = slice(None)
        else:
            row_per_process = rows // comm.size
            s = slice(comm.rank * row_per_process, (comm.rank + 1) * row_per_process)

        bg_ra = file["RA"][s]
        bg_dec = file["DEC"][s]
        g1_values = file["G1"][s]
        g2_values = file["G2"][s]

        # ========= SIGN-FLIP ==========
        if flip_g1:
            g1_values = -g1_values
        if flip_g2:
            g2_values = -g2_values
        # ==============================

        weights = file["weight"][s] if "weight" in file else np.ones_like(bg_ra)

    valid_mask = np.isfinite(bg_ra) & np.isfinite(bg_dec) & np.isfinite(g1_values) & np.isfinite(g2_values) & np.isfinite(weights)
    bg_ra, bg_dec, g1_values, g2_values, weights = bg_ra[valid_mask], bg_dec[valid_mask], g1_values[valid_mask], g2_values[valid_mask], weights[valid_mask]
    bg_coords = np.radians(np.column_stack((bg_ra, bg_dec)))

    max_distance = 0
    bin_sums_plus = np.zeros(num_bins)
    bin_sums_cross = np.zeros(num_bins)
    bin_weighted_distances = np.zeros(num_bins)
    bin_weights = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    for label in unique_labels:
        filament_mask = labels == label
        filament_coords = np.radians(np.column_stack((ra_values[filament_mask], dec_values[filament_mask])))

        nbrs = NearestNeighbors(n_neighbors=1, metric="haversine").fit(filament_coords)
        distances, indices = nbrs.kneighbors(bg_coords)
        matched_filament_points = filament_coords[indices[:, 0]]

        delta_ra = matched_filament_points[:, 0] - bg_coords[:, 0]
        delta_dec = matched_filament_points[:, 1] - bg_coords[:, 1]
        phi = np.arctan2(delta_dec, delta_ra * np.cos(bg_coords[:, 1]))

        g_plus = -g1_values * np.cos(2 * phi) + g2_values * np.sin(2 * phi)
        g_cross = g1_values * np.sin(2 * phi) - g2_values * np.cos(2 * phi)

        max_distance = max(max_distance, np.max(distances))

        min_ang_rad = np.radians(1 / 60)       # 1 arcmin
        max_ang_rad = np.radians(1.0)          # 1 degree
        bins = np.logspace(np.log10(min_ang_rad), np.log10(max_ang_rad), num_bins + 1)
		
        bin_indices = np.digitize(distances[:, 0], bins) - 1
        valid_bins = (bin_indices >= 0) & (bin_indices < num_bins)

        np.add.at(bin_sums_plus, bin_indices[valid_bins], weights[valid_bins] * g_plus[valid_bins])
        np.add.at(bin_sums_cross, bin_indices[valid_bins], weights[valid_bins] * g_cross[valid_bins])
        np.add.at(bin_weighted_distances, bin_indices[valid_bins], weights[valid_bins] * distances[valid_bins, 0])
        np.add.at(bin_weights, bin_indices[valid_bins], weights[valid_bins])
        np.add.at(bin_counts, bin_indices[valid_bins], 1)

    sum_in_place(bin_sums_plus, comm)
    sum_in_place(bin_sums_cross, comm)
    sum_in_place(bin_weighted_distances, comm)
    sum_in_place(bin_weights, comm)
    sum_in_place(bin_counts, comm)

    if comm is not None and comm.rank != 0:
        return

    weighted_g_plus = np.divide(bin_sums_plus, bin_weights, out=np.zeros_like(bin_sums_plus), where=bin_weights > 0)
    weighted_g_cross = np.divide(bin_sums_cross, bin_weights, out=np.zeros_like(bin_sums_cross), where=bin_weights > 0)
    weighted_real_distances = np.divide(bin_weighted_distances, bin_weights, out=np.zeros_like(bin_weighted_distances), where=bin_weights > 0)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    output_data = np.column_stack((bin_centers, weighted_real_distances, weighted_g_plus, weighted_g_cross, bin_counts, bin_weights))
    np.savetxt(output_shear_file, output_data, delimiter=",", header="Bin_Center,Weighted_Real_Distance,Weighted_g_plus,Weighted_g_cross,Counts,bin_weight", comments='')

    print(f"Shear processing completed in {time.time() - start_time:.2f} seconds.")

    # === PLOTTING IN LOG-LOG ===
    arcmin_centers = np.degrees(bin_centers) * 60
    plt.figure()
    plt.loglog(arcmin_centers, np.abs(weighted_g_plus), marker='o', label='|g_plus|')
    plt.loglog(arcmin_centers, np.abs(weighted_g_cross), marker='x', label='|g_cross|')
    plt.xlabel("Separation (arcmin)")
    plt.ylabel("Shear amplitude")
    plt.title("Tangential and Cross Shear")
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plot_file = output_shear_file.replace(".csv", "_shear_plot.png")
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f"Saved shear plot: {plot_file}")
    # ========================

filament_dir = "example/filaments"
os.makedirs(filament_dir, exist_ok=True)

final_percentiles = [0, 10, 25, 40, 50, 60, 75, 85, 90, 95]
for fp in final_percentiles:
    if comm is None or comm.rank == 0:
        print(f"[rank 0] Processing filaments for final_percentile={fp}")
        h5_file = f"example/Ridges_final_p{fp:02d}/ridges_p{fp:02d}.h5"
        with h5py.File(h5_file, 'r') as f:
            Ridges = f["ridges"][:]

        mst = build_mst(Ridges)
        branch_points = detect_branch_points(mst)
        filament_segments = split_mst_at_branches(mst, branch_points)
        filament_labels = segment_filaments_with_dbscan(Ridges, filament_segments)

        filament_h5 = os.path.join(filament_dir, f"filaments_p{fp:02d}.h5")
        save_filaments_to_hdf5(Ridges, filament_labels, filament_h5)

    if comm is not None:
        comm.Barrier()

    # Shear processing (all ranks)
    filament_h5 = os.path.join(filament_dir, f"filaments_p{fp:02d}.h5")
    shear_csv = os.path.join(filament_dir, f"shear_p{fp:02d}.csv")

    # Run with normal signs
    #process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv)

    # ===== Test G1/G2 sign flips =======
	
    # process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv.replace(".csv", "_flipG1.csv"), flip_g1=True)
    # process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv.replace(".csv", "_flipG2.csv"), flip_g2=True)
    process_shear_sims(filament_h5, BG_data, output_shear_file=shear_csv.replace(".csv", "_flipG1G2.csv"), flip_g1=True, flip_g2=True)
