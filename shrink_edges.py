import os
import numpy as np
import h5py
import healpy as hp
import matplotlib.pyplot as plt
import time

# --- start timer ---
t0 = time.time()


# --- config ---
mask_filename = "des-data/desy3_gold_mask.npy"
ridge_file = "example_zl04_mesh5e5/Ridges_final_p15/ridges_p15.h5"
output_dir = "example_zl04_mesh5e5/shrinked_ridges"

os.makedirs(output_dir, exist_ok=True)

# --- parameters ---
nside = 512
arcmin = np.pi / 180.0 / 60.0
radius_arcmin = 4.0          # radius of the disk filter
min_coverage = 0.9           # minimum coverage fraction allowed (0–1)


def load_mask(mask_filename, nside):
    """
    Loads and ud_grades a binary mask from an input numpy file.
    """
    input_mask_nside = 4096
    hit_pix = np.load(mask_filename)
    mask = np.zeros(hp.nside2npix(input_mask_nside))
    mask[hit_pix] = 1
    mask = hp.reorder(mask, n2r=True)
    mask = hp.ud_grade(mask, nside_out=nside)
    return mask
    

def ridge_edge_filter_disk(ridge_ra, ridge_dec, mask, nside, radius_arcmin, min_coverage=1.0):
    """
    Ridge filter: keep ridge points only if coverage inside a disk of given radius
    is >= min_coverage.

    Args:
        ridge_ra (np.ndarray): RA [deg]
        ridge_dec (np.ndarray): Dec [deg]
        mask (np.ndarray): Healpix mask (binary)
        nside (int): nside of mask
        radius_arcmin (float): disk radius [arcmin]
        min_coverage (float): minimum required fraction of valid pixels (0-1)

    Returns:
        np.ndarray (bool): keep mask for ridge points
    """
    radius = np.radians(radius_arcmin / 60.0)  # arcmin → radians

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


# --- load mask ---
print("Loading mask...")
mask = load_mask(mask_filename, nside)

# --- load ridge file ---
print("Loading ridge HDF5...")
try:
    with h5py.File(ridge_file, "r") as f:
        ridges = f["ridges"][:]  # shape (N, 2) = (dec, ra)
except FileNotFoundError:
    print(f"Error: The file '{ridge_file}' was not found.")
    ridges = None

if ridges is None:
    print("Exiting due to file loading error.")
    exit()

ridge_dec = ridges[:, 0]
ridge_ra = ridges[:, 1]
n_total = len(ridges)
print(f"Loaded {n_total} ridges")

# --- apply filter ---
print("Applying ridge filter...")
keep_idx = ridge_edge_filter_disk(ridge_ra, ridge_dec, mask, nside,
                                  radius_arcmin=radius_arcmin,
                                  min_coverage=min_coverage)

ridges_clean = ridges[keep_idx]
print(f"Kept {len(ridges_clean)} ridges (fraction = {len(ridges_clean)/n_total:.2f})")

# --- save output ---
out_file = os.path.join(output_dir, "ridges_p15_shrinked.h5")
with h5py.File(out_file, "w") as f:
    f.create_dataset("ridges", data=ridges_clean)

print(f"Saved cleaned ridges to {out_file}")

# --- end timer ---
t1 = time.time()
print(f"Total execution time: {t1 - t0:.2f} seconds")

# --- plot diagnostic ---
ridges_ra_deg = ridges[:, 1]
ridges_dec_deg = ridges[:, 0]
ridges_clean_ra_deg = ridges_clean[:, 1]
ridges_clean_dec_deg = ridges_clean[:, 0]

plt.figure(figsize=(8, 6))
plt.scatter(ridges_ra_deg, ridges_dec_deg, s=1, alpha=0.3, label="All ridges")
plt.scatter(ridges_clean_ra_deg, ridges_clean_dec_deg, s=1, alpha=0.6, label="Filtered ridges")
plt.xlabel("RA")
plt.ylabel("Dec")
plt.title(f"Ridges before/after filtering\nradius={radius_arcmin} arcmin, min_coverage={min_coverage}")
plt.legend()
plt.tight_layout()

plot_file = os.path.join(output_dir, "ridges_filter_diagnostic.png")
plt.savefig(plot_file, dpi=200)
plt.close()

print(f" Saved diagnostic plot to {plot_file}")
 