"""
Shrink ridges near survey boundaries for all runs in:
simulation_ridges_comparative_analysis/<base>/<band_X.X>/Ridges_final_p15

Creates:
simulation_ridges_comparative_analysis/<base>/<band_X.X>/Shrinked_Ridges_final_p15/
"""

import os
import numpy as np
import h5py
import healpy as hp
import matplotlib.pyplot as plt
import time

# ======================================================
# === CONFIGURATION ===
# ======================================================

mask_filename = "des-data/desy3_gold_mask.npy"   # mask file
base_root = "simulation_ridges_comparative_analysis"  # main simulation folder
radius_arcmin = 4.0          # disk filter radius
min_coverage = 0.9           # fraction of mask pixels required
nside = 512                  # map resolution

# ======================================================
# === FUNCTIONS ===
# ======================================================

def load_mask(mask_filename, nside):
    """Load and downgrade a binary mask from .npy list of hit pixels."""
    input_mask_nside = 4096
    hit_pix = np.load(mask_filename)
    mask = np.zeros(hp.nside2npix(input_mask_nside))
    mask[hit_pix] = 1
    mask = hp.reorder(mask, n2r=True)
    mask = hp.ud_grade(mask, nside_out=nside)
    return mask


def ridge_edge_filter_disk(ridge_ra, ridge_dec, mask, nside, radius_arcmin, min_coverage=1.0):
    """Return boolean array of ridge points with coverage ≥ min_coverage."""
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


def process_ridge_file(ridge_file, mask, nside, radius_arcmin, min_coverage, output_dir):
    """Apply the shrink filter to one ridge file."""
    with h5py.File(ridge_file, "r") as f:
        ridges = f["ridges"][:]
    ridge_dec = ridges[:, 0]
    ridge_ra = ridges[:, 1]
    n_total = len(ridges)

    keep_idx = ridge_edge_filter_disk(
        ridge_ra, ridge_dec, mask, nside,
        radius_arcmin=radius_arcmin, min_coverage=min_coverage
    )
    ridges_clean = ridges[keep_idx]
    print(f"[shrink] {os.path.basename(ridge_file)}: kept {len(ridges_clean)}/{n_total}")

    # Save to output folder
    base_name = os.path.basename(ridge_file).replace(".h5", "_shrinked.h5")
    out_file = os.path.join(output_dir, base_name)
    with h5py.File(out_file, "w") as f:
        f.create_dataset("ridges", data=ridges_clean)

    # Plot diagnostic
    plot_file = out_file.replace(".h5", "_diagnostic.png")
    plt.figure(figsize=(8, 6))
    plt.scatter(ridge_ra, ridge_dec, s=1, alpha=0.3, label="All ridges")
    plt.scatter(ridges_clean[:, 1], ridges_clean[:, 0], s=1, alpha=0.6, label="Filtered ridges")
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title(f"Shrinked ridges\nradius={radius_arcmin} arcmin, min_cov={min_coverage}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    plt.close()

    print(f"[plot] Saved diagnostic → {plot_file}")


# ======================================================
# === MAIN EXECUTION ===
# ======================================================

if __name__ == "__main__":
    t0 = time.time()
    print("=== Ridge shrink processing started ===")

    # Load mask once
    print(f"[load] Mask: {mask_filename}")
    mask = load_mask(mask_filename, nside)

    # Walk over all bases and bands
    for base_label in os.listdir(base_root):
        base_path = os.path.join(base_root, base_label)
        if not os.path.isdir(base_path):
            continue

        for band_folder in os.listdir(base_path):
            if not band_folder.startswith("band_"):
                continue

            band_path = os.path.join(base_path, band_folder)
            ridges_dir = os.path.join(band_path, "Ridges_final_p15")
            if not os.path.exists(ridges_dir):
                continue

            out_dir = os.path.join(band_path, "Shrinked_Ridges_final_p15")
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n[dir] Processing {ridges_dir} → {out_dir}")

            ridge_files = [f for f in os.listdir(ridges_dir) if f.endswith(".h5")]
            for rf in ridge_files:
                full_path = os.path.join(ridges_dir, rf)
                process_ridge_file(
                    ridge_file=full_path,
                    mask=mask,
                    nside=nside,
                    radius_arcmin=radius_arcmin,
                    min_coverage=min_coverage,
                    output_dir=out_dir
                )

    t1 = time.time()
    print(f"\n=== Done. Total execution time: {t1 - t0:.2f} s ===")
