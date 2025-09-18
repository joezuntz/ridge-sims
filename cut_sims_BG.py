import os
import h5py
import numpy as np

base_sim_dir = "lhc_run_sims_50"

for run_id in range(1, 9):  # runs 1 through 8 for now
    bg_data_path = os.path.join(base_sim_dir, f"run_{run_id}", "source_catalog_0.npy")
    output_file_path = os.path.join(base_sim_dir, f"run_{run_id}", "source_catalog_cutzl04.h5")

    if not os.path.exists(bg_data_path):
        print(f"Run {run_id}: Missing input file {bg_data_path}, skipping.")
        continue

    try:
        with h5py.File(bg_data_path, "r") as file:
            bg_ra_full = file["RA"][:]
            bg_dec_full = file["DEC"][:]
            g1_values_full = file["G1"][:]
            g2_values_full = file["G2"][:]
            z_true_full = file["Z_TRUE"][:]
            weights_full = file["weight"][:] if "weight" in file else np.ones_like(bg_ra_full)

        # Apply cuts
        z_mask = z_true_full > 0.4
        valid_mask = (
            np.isfinite(bg_ra_full)
            & np.isfinite(bg_dec_full)
            & np.isfinite(g1_values_full)
            & np.isfinite(g2_values_full)
            & np.isfinite(weights_full)
            & z_mask
        )

        # Filtered arrays
        bg_ra_filtered = bg_ra_full[valid_mask]
        bg_dec_filtered = bg_dec_full[valid_mask]
        g1_values_filtered = g1_values_full[valid_mask]
        g2_values_filtered = g2_values_full[valid_mask]
        z_true_filtered = z_true_full[valid_mask]
        weights_filtered = weights_full[valid_mask]

        # Save to new HDF5
        with h5py.File(output_file_path, "w") as hf:
            hf.create_dataset("RA", data=bg_ra_filtered)
            hf.create_dataset("DEC", data=bg_dec_filtered)
            hf.create_dataset("G1", data=g1_values_filtered)
            hf.create_dataset("G2", data=g2_values_filtered)
            hf.create_dataset("Z_TRUE", data=z_true_filtered)
            hf.create_dataset("weight", data=weights_filtered)

        print(f"Run {run_id}: Filtered data saved to {output_file_path}")
        print(f"   Original size: {len(bg_ra_full)} → Filtered size: {len(bg_ra_filtered)}")

    except Exception as e:
        print(f"Run {run_id}: Failed due to {e}")
