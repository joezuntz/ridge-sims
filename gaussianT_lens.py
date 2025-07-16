import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

# --- Configuration ---
mock_lens_output_dir = "mock_lens_data"
mock_lens_filename = "mock_t_shape_gaussian_lens.hdf5" # New filename for this version

# --- T-Shape Parameters ---
t_center_ra = 180.0  # Central RA for the T (in the middle of 0-360 range)
t_center_dec = 0.0   # Central DEC for the T
t_size = 20.0        # Overall length/width scale of the T

# Gaussian density parameters for the T's width
gaussian_sigma = 0.5 # INCREASED: Standard deviation of the Gaussian 
# To make the T lines significantly wider and the density gradient more visible.
points_per_cross_section = 50 # INCREASED: Number of points to sample across the Gaussian width for each 'slice'

num_slices_per_segment = 150 # INCREASED: Number of 'slices' along the length of each T segment
# To maintain continuity for longer, wider lines.

# Calculated total T points: 2 segments * num_slices_per_segment * points_per_cross_section
# e.g., 2 * 150 * 50 = 15000 T-shape points

# --- Noise Parameters ---
noise_area_ra_min = t_center_ra - 20.0 
noise_area_ra_max = t_center_ra + 20.0
noise_area_dec_min = t_center_dec - 20.0
noise_area_dec_max = t_center_dec + 20.0
num_noise_points = 1000 # Increased number of noise points, still much less dense than T


# Create output directory
if not os.path.exists(mock_lens_output_dir):
    os.makedirs(mock_lens_output_dir)

# --- Generate T-Shape Coordinates with Gaussian Density ---
t_ra = []
t_dec = []

# 1. Vertical bar of the T
base_decs_vert = np.linspace(t_center_dec - 0.7 * t_size, t_center_dec + 0.7 * t_size, num_slices_per_segment)
for base_dec in base_decs_vert:
    sampled_ras = np.random.normal(loc=t_center_ra, scale=gaussian_sigma, size=points_per_cross_section)
    t_ra.extend(sampled_ras)
    t_dec.extend(np.full_like(sampled_ras, base_dec))

# 2. Horizontal bar of the T (at the top of the vertical bar)
base_ras_horiz = np.linspace(t_center_ra - 0.7 * t_size, t_center_ra + 0.7 * t_size, num_slices_per_segment)
for base_ra in base_ras_horiz:
    sampled_decs = np.random.normal(loc=t_center_dec + 0.7 * t_size, scale=gaussian_sigma, size=points_per_cross_section)
    t_ra.extend(np.full_like(sampled_decs, base_ra))
    t_dec.extend(sampled_decs)

t_ra = np.array(t_ra)
t_dec = np.array(t_dec)

# --- Generate Noise Coordinates (uniform distribution) ---
noise_ra = np.random.uniform(noise_area_ra_min, noise_area_ra_max, num_noise_points)
noise_dec = np.random.uniform(noise_area_dec_min, noise_area_dec_max, num_noise_points)

# --- Combine T-Shape and Noise ---
all_ra = np.concatenate((t_ra, noise_ra))
all_dec = np.concatenate((t_dec, noise_dec))

# --- Ensure RA is within 0-360 range and normalized ---
all_ra = (all_ra + 360) % 360

# --- Save to HDF5 ---
output_filepath = os.path.join(mock_lens_output_dir, mock_lens_filename)

try:
    with h5py.File(output_filepath, 'w') as f:
        filtered_catalog_group = f.create_group("filtered_catalog")
        filtered_catalog_group.create_dataset("ra", data=all_ra)
        filtered_catalog_group.create_dataset("dec", data=all_dec)
    print(f"Mock lens data with longer, wider Gaussian T saved to {output_filepath}")

    # --- Visualize the generated data ---
    plt.figure(figsize=(12, 10))
    plt.scatter(all_ra, all_dec, s=5, alpha=0.5, label='Combined Mock Data')
    plt.scatter(t_ra, t_dec, s=10, alpha=0.8, color='red', label='T-Shape Structure (Gaussian Density)')

    plt.title("Generated Mock T-Shape Lens (Longer, Wider Gaussian) with Random Noise")
    plt.xlabel("Right Ascension (RA)")
    plt.ylabel("Declination (DEC)")
    plt.gca().invert_xaxis() # Invert RA axis
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plot_filename = os.path.join(mock_lens_output_dir, "mock_t_shape_gaussian_lens_longer_wider_plot.png")
    plt.savefig(plot_filename)
    print(f"Visualization saved to {plot_filename}")
    plt.show()

except Exception as e:
    print(f"Error saving mock lens data or generating plot: {e}")