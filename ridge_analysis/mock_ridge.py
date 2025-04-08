import numpy as np
import h5py
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
from tools import *



def generate_artificial_background(output_ridge_file, output_bg_file,
                                   ridge_ra_center=150, ridge_dec_center=30,
                                   ridge_length=5, num_ridge_points=10000,
                                   num_bg_points=1000, ra_range=2, dec_range=2):
    """
    Generates a synthetic straight filament (ridge) and a random field of background galaxies.
    - The ridge is a straight dense diagonal line in RA-DEC.
    - Background galaxies are randomly placed in a rectangular patch.
    - Shear field:
        g1 = 0 everywhere,
        g2 = 1 / (perpendicular distance to ridge line), clipped at small distances.
    - Data is saved in HDF5 format.
    """

    # Set random seed for reproducibility
    np.random.seed(42)

    # --- RIDGE SETUP ---

    # Half-lengths in RA and DEC (ridge will go from -d to +d centered at the ridge center)
    d_ra = ridge_length / 2
    d_dec = ridge_length / 2

    # Ridge is a diagonal line: slope = d_dec / d_ra = 1 (45°)
    # Generate a linearly spaced set of points for the ridge
    ridge_ra = np.linspace(ridge_ra_center - d_ra, ridge_ra_center + d_ra, num_ridge_points)
    ridge_dec = np.linspace(ridge_dec_center - d_dec, ridge_dec_center + d_dec, num_ridge_points)
    ridge_positions = np.column_stack((ridge_ra, ridge_dec))  # Shape (num_ridge_points, 2)

    # --- BACKGROUND SETUP ---

    # Random background positions in a box around the ridge center
    bg_ra = np.random.uniform(ridge_ra_center - ra_range, ridge_ra_center + ra_range, num_bg_points)
    bg_dec = np.random.uniform(ridge_dec_center - dec_range, ridge_dec_center + dec_range, num_bg_points)

    # --- DISTANCE TO RIDGE LINE (GEOMETRICALLY) ---

    # Direction vector of the ridge line
    v = np.array([d_ra, d_dec])

    # Normalize to get unit direction vector
    v_norm = v / np.linalg.norm(v)

    # Ridge center treated as a point on the infinite line
    r0 = np.array([ridge_ra_center, ridge_dec_center])

    # Compute vector from ridge center to each background point
    r_vecs = np.column_stack((bg_ra, bg_dec)) - r0  # Shape (num_bg_points, 2)

    # Project each vector onto the ridge direction
    proj_lengths = np.dot(r_vecs, v_norm)  # Scalar projection lengths along the ridge
    proj_points = np.outer(proj_lengths, v_norm)  # Projected points along the line

    # Perpendicular vectors from the line to the background points
    perp_vecs = r_vecs - proj_points

    # Distance is just the norm of the perpendicular vector
    perp_distances = np.linalg.norm(perp_vecs, axis=1)

    # --- SHEAR FIELD ---

    # Minimum allowed distance to avoid divergence in g2
    min_distance = 0.01
    safe_distances = np.clip(perp_distances, min_distance, None)

    # g2 = 1 / distance, clipped at small distances to avoid numerical issues
    g2_array = 1.0 / safe_distances

    # g1 = 0 everywhere by construction
    g1_array = np.zeros_like(g2_array)

    # --- SAVE DATA ---

    # Save ridge points (as 1D RA and DEC arrays)
    with h5py.File(output_ridge_file, 'w') as f:
        f.create_dataset('RA', data=ridge_ra)
        f.create_dataset('DEC', data=ridge_dec)

    # Save background galaxy data and shear
    with h5py.File(output_bg_file, 'w') as f:
        bg_group = f.create_group("background")
        bg_group.create_dataset('ra', data=bg_ra)
        bg_group.create_dataset('dec', data=bg_dec)
        bg_group.create_dataset('g1', data=g1_array)
        bg_group.create_dataset('g2', data=g2_array)
        bg_group.create_dataset('distance_to_ridge', data=perp_distances)
        bg_group.create_dataset('weight', data=np.ones(num_bg_points))  # Uniform weights

    # Print a preview of the shear values
    print("✅ Ridge and background generated.")
    print("First 5 g2 values:", g2_array[:5])

        
# Generate test data
generate_artificial_background('test_ridge.h5', 'test_background.h5')

def process_shear0(filament_file, bg_data, output_shear_file, k=1, num_bins=20):
    """Compute shear transformation and bin results, saving to output file with final plots."""
    start_time = time.time()

    # Load filament data
    with h5py.File(filament_file, "r") as hdf:
        dataset = hdf["data"]
        ra_values = dataset["RA"][:]
        dec_values = dataset["DEC"][:]
        labels = dataset["Filament_Label"][:]

    unique_labels = np.unique(labels)

    # Load background data
    with h5py.File(bg_data, "r") as file:
        background_group = file["background"]
        bg_ra = background_group["ra"][:]
        bg_dec = background_group["dec"][:]
        g1_values = background_group["g1"][:]
        g2_values = background_group["g2"][:]
        weights = background_group["weight"][:]

    # Filter valid background points
    valid_mask = np.isfinite(bg_ra) & np.isfinite(bg_dec) & np.isfinite(g1_values) & np.isfinite(g2_values) & np.isfinite(weights)
    bg_ra, bg_dec, g1_values, g2_values, weights = bg_ra[valid_mask], bg_dec[valid_mask], g1_values[valid_mask], g2_values[valid_mask], weights[valid_mask]
    bg_coords = np.radians(np.column_stack((bg_ra, bg_dec)))

    # Initialize bins
    max_distance = 0
    bin_sums_plus = np.zeros(num_bins)
    bin_sums_cross = np.zeros(num_bins)
    bin_weighted_distances = np.zeros(num_bins)
    bin_weights = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    # Process each filament
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

        # Update bins
        bins = np.linspace(0, max_distance * 1.05, num_bins + 1)
        bin_indices = np.digitize(distances[:, 0], bins) - 1
        valid_bins = (bin_indices >= 0) & (bin_indices < num_bins)

        np.add.at(bin_sums_plus, bin_indices[valid_bins], weights[valid_bins] * g_plus[valid_bins])
        np.add.at(bin_sums_cross, bin_indices[valid_bins], weights[valid_bins] * g_cross[valid_bins])
        np.add.at(bin_weighted_distances, bin_indices[valid_bins], weights[valid_bins] * distances[valid_bins, 0])
        np.add.at(bin_weights, bin_indices[valid_bins], weights[valid_bins])
        np.add.at(bin_counts, bin_indices[valid_bins], 1)

    # Final bin averages
    weighted_g_plus = np.divide(bin_sums_plus, bin_weights, out=np.zeros_like(bin_sums_plus), where=bin_weights > 0)
    weighted_g_cross = np.divide(bin_sums_cross, bin_weights, out=np.zeros_like(bin_sums_cross), where=bin_weights > 0)
    weighted_real_distances = np.divide(bin_weighted_distances, bin_weights, out=np.zeros_like(bin_weighted_distances), where=bin_weights > 0)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    output_data = np.column_stack((bin_centers, weighted_real_distances, weighted_g_plus, weighted_g_cross, bin_counts, bin_weights))
    np.savetxt(output_shear_file, output_data, delimiter=",", header="Bin_Center,Weighted_Real_Distance,Weighted_g_plus,Weighted_g_cross,Counts,bin_weight", comments='')

    print(f"Shear processing completed in {time.time() - start_time:.2f} seconds.")

    # ========== Final Plots ==========
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

        # Save plot
        plt.savefig(filename)
        print(f"Saved plot to: {filename}")
        plt.close()

    # Optional: set custom directory to store plots
    plot_dir = "shear_plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Save linear scale plots
    plot_results(
        weighted_real_distances, weighted_g_plus, "g+", "Shear g+ vs Distance",
        os.path.join(plot_dir, "linear_g_plus_vs_distance.png")
    )
    plot_results(
        weighted_real_distances, weighted_g_cross, "g×", "Shear g× vs Distance",
        os.path.join(plot_dir, "linear_g_cross_vs_distance.png")
    )

    # Save log-log scale plots
    plot_results(
        weighted_real_distances, np.abs(weighted_g_plus), "|g+|", "Log-Log Plot: |g+| vs Distance",
        os.path.join(plot_dir, "loglog_abs_g_plus_vs_distance.png"),
        loglog=True
    )
    plot_results(
        weighted_real_distances, np.abs(weighted_g_cross), "|g×|", "Log-Log Plot: |g×| vs Distance",
        os.path.join(plot_dir, "loglog_abs_g_cross_vs_distance.png"),
        loglog=True
    )


# --- saving filament ---
def save_single_filament_to_hdf5(ridge_ra, ridge_dec, filename, dataset_name="data"):
    labels = np.ones(len(ridge_ra), dtype=np.int64)
    dtype = [("RA", "f8"), ("DEC", "f8"), ("Filament_Label", "i8")]
    structured_data = np.array(list(zip(ridge_ra, ridge_dec, labels)), dtype=dtype)
    with h5py.File(filename, "w") as hdf:
        hdf.create_dataset(dataset_name, data=structured_data)

# Example usage
Ridges = 'test_ridge.h5'
BG_data = "test_background.h5"
with h5py.File(Ridges, 'r') as f:
    ridge_ra = f['RA'][:]
    ridge_dec = f['DEC'][:]

save_single_filament_to_hdf5(ridge_ra, ridge_dec, "filament_outputs/test_Filament_label.hdf5")
print("Saved segmented filament results")

# Run the shear processing function
process_shear0("filament_outputs/test_Filament_label.hdf5", BG_data, "output_shear.csv", num_bins=20)

