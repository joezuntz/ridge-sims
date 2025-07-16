import os
import numpy as np
import h5py

base_sim_dir = "lhc_run_sims"
output_dir = "little_runs_adaptive"
target_points = 300
square_size = 5    # Still considering a 5x5 degree area for the initial check
max_shift = 5      # Maximum number of degrees to shift in each direction

# Define the initial base boundaries for the 5x5 degree square
# These are the fixed starting points for the shifting logic
ra_min_base = 215
ra_max_base = 220
dec_min_base = -45
dec_max_base = -40

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def extract_square(coordinates, ra_min, ra_max, dec_min, dec_max, num_points):
    ra_mask = (coordinates[:, 1] >= ra_min) & (coordinates[:, 1] <= ra_max)
    dec_mask = (coordinates[:, 0] >= dec_min) & (coordinates[:, 0] <= dec_max)
    selection_mask = ra_mask & dec_mask
    selected_coordinates = coordinates[selection_mask]
    if selected_coordinates.shape[0] >= num_points:
        return selected_coordinates[:num_points]
    return None

for run_id in range(1, 11):  
    run_dir = os.path.join(base_sim_dir, f"run_{run_id}")
    source_catalog_file = os.path.join(run_dir, "lens_catalog_0.npy")
    output_file_prefix = os.path.join(output_dir, f"little_run_{run_id}")

    print(f"Processing Run {run_id}...")

    try:
        with h5py.File(source_catalog_file, 'r') as f:
            ra = f["RA"][:]
            ra = (ra + 180) % 360
            dec = f["DEC"][:]
            coordinates = np.column_stack((dec, ra))

        # The initial_ra_center and initial_dec_center are now effectively derived
        # from ra_min_base, ra_max_base, etc.
        # These lines are kept for clarity if we need a conceptual center,
        # but the actual shifting uses the base min/max.
        # initial_ra_center = (ra_min_base + ra_max_base) / 2
        # initial_dec_center = (dec_min_base + dec_max_base) / 2

        found_square = False
        for ra_shift in np.arange(-max_shift, max_shift + 1, 1):
            for dec_shift in np.arange(-max_shift, max_shift + 1, 1):
                # Use the base boundaries for shifting
                current_ra_min = ra_min_base + ra_shift
                current_ra_max = ra_max_base + ra_shift
                current_dec_min = dec_min_base + dec_shift
                current_dec_max = dec_max_base + dec_shift

                extracted_points = extract_square(coordinates, current_ra_min, current_ra_max, current_dec_min, current_dec_max, target_points)

                if extracted_points is not None:
                    output_file = f"{output_file_prefix}_ra_{current_ra_min:.1f}_{current_ra_max:.1f}_dec_{current_dec_min:.1f}_{current_dec_max:.1f}.npy"
                    np.save(output_file, extracted_points)
                    print(f"  Found and saved {target_points} points in shifted square (RA: {current_ra_min:.1f}-{current_ra_max:.1f}, DEC: {current_dec_min:.1f}-{current_dec_max:.1f}) to {output_file}")
                    found_square = True
                    break  # Move to the next run once a suitable square is found
            if found_square:
                break

        if not found_square:
            print(f"  Could not find a {square_size}x{square_size} degree region with at least {target_points} points within the shift range for Run {run_id}.")

    except FileNotFoundError:
        print(f"  Error: Source catalog file not found at: {source_catalog_file}")
    except Exception as e:
        print(f"  Error processing Run {run_id}: {e}")
        print(e) # Print the error for debugging

print("\n--- Finished processing all runs ---")