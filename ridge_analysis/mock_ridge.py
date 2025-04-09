import numpy as np
import h5py
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import os
import time
from tools import *


def generate_artificial_background(output_ridge_file, output_bg_file,
                                   ridge_ra_center=150, ridge_dec_center=30,
                                   ridge_length=5, num_ridge_points=10000,
                                   num_bg_points=1000, ra_range=2, dec_range=2):
    """
    Generates a synthetic straight filament (ridge) and a random field of background galaxies.
    All positions and angles are in radians.
    """
    # Convert all angular inputs to radians
    ridge_ra_center = np.radians(ridge_ra_center)
    ridge_dec_center = np.radians(ridge_dec_center)
    ridge_length = np.radians(ridge_length)
    ra_range = np.radians(ra_range)
    dec_range = np.radians(dec_range)

    # Ridge endpoints and linear spacing
    d_ra = ridge_length / 2
    d_dec = ridge_length / 2
    ridge_ra = np.linspace(ridge_ra_center - d_ra, ridge_ra_center + d_ra, num_ridge_points)
    ridge_dec = np.linspace(ridge_dec_center - d_dec, ridge_dec_center + d_dec, num_ridge_points)
    ridge_positions = np.column_stack((ridge_ra, ridge_dec))

    # Random background galaxy positions
    bg_ra = np.random.uniform(ridge_ra_center - ra_range, ridge_ra_center + ra_range, num_bg_points)
    bg_dec = np.random.uniform(ridge_dec_center - dec_range, ridge_dec_center + dec_range, num_bg_points)

    # Ridge direction vector
    v = np.array([d_ra, d_dec])
    v_norm = v / np.linalg.norm(v)

    # Vector from ridge center to each background galaxy
    r0 = np.array([ridge_ra_center, ridge_dec_center])
    r_vecs = np.column_stack((bg_ra, bg_dec)) - r0

    # Project r_vecs onto ridge direction to get parallel component
    proj_lengths = np.dot(r_vecs, v_norm)
    proj_points = np.outer(proj_lengths, v_norm)
    perp_vecs = r_vecs - proj_points
    perp_distances = np.linalg.norm(perp_vecs, axis=1)

    # Compute angles from ridge center to each background galaxy
    delta_ra = bg_ra - ridge_ra_center
    delta_dec = bg_dec - ridge_dec_center
    angles = np.arctan2(delta_dec, delta_ra)  # φ in radians

    # Assign tangential shear: g_+ = 1 / R, g_x = 0
    min_distance = np.radians(0.01)
    safe_distances = np.clip(perp_distances, min_distance, None)
    g_plus = 1.0 / safe_distances
    g_cross = np.zeros_like(g_plus)

    # Rotate (g_+, g_x) into (g1, g2)
    g1_array = -g_plus * np.cos(2 * angles)
    g2_array = -g_plus * np.sin(2 * angles)

    # Save ridge to HDF5
    with h5py.File(output_ridge_file, 'w') as f:
        f.create_dataset('RA', data=np.degrees(ridge_ra))
        f.create_dataset('DEC', data=np.degrees(ridge_dec))

    # Save background galaxies and shear
    with h5py.File(output_bg_file, 'w') as f:
        bg_group = f.create_group("background")
        bg_group.create_dataset('ra', data=np.degrees(bg_ra))
        bg_group.create_dataset('dec', data=np.degrees(bg_dec))
        bg_group.create_dataset('g1', data=g1_array)
        bg_group.create_dataset('g2', data=g2_array)
        bg_group.create_dataset('distance_to_ridge', data=np.degrees(perp_distances))
        bg_group.create_dataset('weight', data=np.ones(num_bg_points))

    print(f"Saved {num_ridge_points} ridge points and {num_bg_points} background galaxies.")

def plot_shear_results(weighted_real_distances, weighted_g_plus, weighted_g_cross, plot_dir="shear_plots"):
    def plot_results(x, y, ylabel, title, filename, loglog=False):
        plt.figure()
        if loglog:
            plt.loglog(x, y, 'o-', label=ylabel)
        else:
            plt.plot(x, y, 'o-', label=ylabel)
        plt.xlabel("Distance (Bin Centers)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved plot to: {filename}")
        plt.close()

    os.makedirs(plot_dir, exist_ok=True)

    plot_results(weighted_real_distances, weighted_g_plus, "g+", "Shear g+ vs Distance", os.path.join(plot_dir, "linear_g_plus_vs_distance.png"))
    plot_results(weighted_real_distances, weighted_g_cross, "g×", "Shear g× vs Distance", os.path.join(plot_dir, "linear_g_cross_vs_distance.png"))
    plot_results(weighted_real_distances, np.abs(weighted_g_plus), "|g+|", "Log-Log Plot: |g+| vs Distance", os.path.join(plot_dir, "loglog_abs_g_plus_vs_distance.png"), loglog=True)
    plot_results(weighted_real_distances, np.abs(weighted_g_cross), "|g×|", "Log-Log Plot: |g×| vs Distance", os.path.join(plot_dir, "loglog_abs_g_cross_vs_distance.png"), loglog=True)


def process_shear0(filament_file, bg_data, output_shear_file, k=1, num_bins=20):
    start_time = time.time()

    with h5py.File(filament_file, "r") as hdf:
        dataset = hdf["data"]
        ra_values = dataset["RA"][:]
        dec_values = dataset["DEC"][:]
        labels = dataset["Filament_Label"][:]

    unique_labels = np.unique(labels)

    with h5py.File(bg_data, "r") as file:
        background_group = file["background"]
        bg_ra = background_group["ra"][:]
        bg_dec = background_group["dec"][:]
        g1_values = background_group["g1"][:]
        g2_values = background_group["g2"][:]
        weights = background_group["weight"][:]

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
        phi = np.arctan2(delta_dec, delta_ra)
        g_plus = -g1_values * np.cos(2 * phi) + g2_values * np.sin(2 * phi)
        g_cross = g1_values * np.sin(2 * phi) - g2_values * np.cos(2 * phi)
        max_distance = max(max_distance, np.max(distances))
        bins = np.linspace(0, max_distance * 1.05, num_bins + 1)
        bin_indices = np.digitize(distances[:, 0], bins) - 1
        valid_bins = (bin_indices >= 0) & (bin_indices < num_bins)
        np.add.at(bin_sums_plus, bin_indices[valid_bins], weights[valid_bins] * g_plus[valid_bins])
        np.add.at(bin_sums_cross, bin_indices[valid_bins], weights[valid_bins] * g_cross[valid_bins])
        np.add.at(bin_weighted_distances, bin_indices[valid_bins], weights[valid_bins] * distances[valid_bins, 0])
        np.add.at(bin_weights, bin_indices[valid_bins], weights[valid_bins])
        np.add.at(bin_counts, bin_indices[valid_bins], 1)

    weighted_g_plus = np.divide(bin_sums_plus, bin_weights, out=np.zeros_like(bin_sums_plus), where=bin_weights > 0)
    weighted_g_cross = np.divide(bin_sums_cross, bin_weights, out=np.zeros_like(bin_sums_cross), where=bin_weights > 0)
    weighted_real_distances = np.divide(bin_weighted_distances, bin_weights, out=np.zeros_like(bin_weighted_distances), where=bin_weights > 0)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    output_data = np.column_stack((bin_centers, weighted_real_distances, weighted_g_plus, weighted_g_cross, bin_counts, bin_weights))
    np.savetxt(output_shear_file, output_data, delimiter=",", header="Bin_Center,Weighted_Real_Distance,Weighted_g_plus,Weighted_g_cross,Counts,bin_weight", comments='')

    print(f"Shear processing completed in {time.time() - start_time:.2f} seconds.")

    plot_shear_results(weighted_real_distances, weighted_g_plus, weighted_g_cross)


def save_single_filament_to_hdf5(ridge_ra, ridge_dec, filename, dataset_name="data"):
    labels = np.ones(len(ridge_ra), dtype=np.int64)
    dtype = [("RA", "f8"), ("DEC", "f8"), ("Filament_Label", "i8")]
    structured_data = np.array(list(zip(ridge_ra, ridge_dec, labels)), dtype=dtype)
    with h5py.File(filename, "w") as hdf:
        hdf.create_dataset(dataset_name, data=structured_data)

# Example usage
generate_artificial_background('test_ridge.h5', 'test_background.h5')
Ridges = 'test_ridge.h5'
BG_data = "test_background.h5"
with h5py.File(Ridges, 'r') as f:
    ridge_ra = f['RA'][:]
    ridge_dec = f['DEC'][:]
save_single_filament_to_hdf5(ridge_ra, ridge_dec, "filament_outputs/test_Filament_label.hdf5")
print("Saved segmented filament results")
process_shear0("filament_outputs/test_Filament_label.hdf5", BG_data, "output_shear.csv", num_bins=20)