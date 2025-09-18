import os
import numpy as np
import h5py
import healpy as hp
import time

# --- configuration ---
mask_filename = "des-data/desy3_gold_mask.npy"
nside = 512
radius_arcmin = 4.0
min_coverage = 0.9

def load_mask(mask_filename, nside):
    input_mask_nside = 4096
    hit_pix = np.load(mask_filename)
    mask = np.zeros(hp.nside2npix(input_mask_nside))
    mask[hit_pix] = 1
    mask = hp.reorder(mask, n2r=True)
    mask = hp.ud_grade(mask, nside_out=nside)
    return mask

def ridge_edge_filter_disk(ridge_ra, ridge_dec, mask, nside, radius_arcmin, min_coverage):
    radius = np.radians(radius_arcmin / 60.0)
    theta_ridges = np.radians(90.0 - ridge_dec)
    phi_ridges = np.radians(ridge_ra)
    vec_ridges = hp.ang2vec(theta_ridges, phi_ridges)
    keep_idx = np.zeros(len(ridge_ra), dtype=bool)

    for i, v in enumerate(vec_ridges):
        disk_pix = hp.query_disc(nside, v, radius, inclusive=True)
        if len(disk_pix) == 0:
            frac = 0.0
        else:
            frac = mask[disk_pix].sum() / len(disk_pix)
        if frac >= min_coverage:
            keep_idx[i] = True
    return keep_idx

def main():
    num_runs = 8
    final_percentile = 15
    mask = load_mask(mask_filename, nside)
    
    for run_id in range(1, num_runs + 1):
        print(f"--- Filtering ridges for Run {run_id} ---")
        
        # Define paths
        ridge_dir = f"example30_band0.4/run_{run_id}/ridges"
        output_dir = f"example30_band0.4/8test/run_{run_id}/ridges_filtered"
        os.makedirs(output_dir, exist_ok=True)
        
        ridge_file = os.path.join(ridge_dir, f"ridges_p{final_percentile:02d}.h5")
        out_file = os.path.join(output_dir, f"ridges_p{final_percentile:02d}_filtered.h5")
        
        try:
            with h5py.File(ridge_file, "r") as f:
                ridges = f["ridges"][:]
            
            ridge_dec = ridges[:, 0]
            ridge_ra = ridges[:, 1]
            n_total = len(ridges)
            
            keep_idx = ridge_edge_filter_disk(ridge_ra, ridge_dec, mask, nside, radius_arcmin, min_coverage)
            ridges_clean = ridges[keep_idx]
            
            with h5py.File(out_file, "w") as f:
                f.create_dataset("ridges", data=ridges_clean)
            
            print(f"  - Kept {len(ridges_clean)} of {n_total} ridges. Saved to {out_file}")

        except FileNotFoundError:
            print(f"  - Error: The file '{ridge_file}' was not found. Skipping.")

if __name__ == "__main__":
    main()