import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
import os

# --- parameters ---
nside = 512
mask_radius_deg = 10.0             # hole radius (deg)
mask_radius = np.radians(mask_radius_deg)  # -> radians
hole_center = (0.0, 0.0)           # (RA, Dec) in radians (0,0)

radius_arcmin = 30.0               # filtering radius
radius = np.radians(radius_arcmin / 60.0)  # arcmin -> rad
min_coverage = 1

output_dir = "example_zl04_mesh5e5/shrinked_ridges"
os.makedirs(output_dir, exist_ok=True)


def make_mock_mask(nside, hole_center, hole_radius):
    npix = hp.nside2npix(nside)
    mask = np.ones(npix)

    ra, dec = hole_center
    theta = np.pi/2 - dec  # colatitude
    phi = ra
    vec = hp.ang2vec(theta, phi)

    hole_pix = hp.query_disc(nside, vec, hole_radius)
    mask[hole_pix] = 0
    return mask


def make_mock_ridges():
    # Straight line in Dec = 0, RA from -30° to +30°
    ra = np.linspace(np.radians(-30), np.radians(30), 200)  # radians
    dec = np.zeros_like(ra)  # radians
    ridges = np.vstack([dec, ra]).T
    return ridges


def ridge_edge_filter_disk(ridge_ra, ridge_dec, mask, nside, radius, min_coverage=1.0):
    theta_ridges = np.pi/2 - ridge_dec
    phi_ridges = ridge_ra
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


# --- build mock data ---
print("Building mock mask...")
mask = make_mock_mask(nside, hole_center, mask_radius)

print("Building mock ridges...")
ridges = make_mock_ridges()
ridge_dec = ridges[:, 0]
ridge_ra = ridges[:, 1]

# --- apply filter ---
print("Applying filter...")
keep_idx = ridge_edge_filter_disk(ridge_ra, ridge_dec, mask, nside,
                                  radius=radius,
                                  min_coverage=min_coverage)
ridges_clean = ridges[keep_idx]
print(f" Kept {len(ridges_clean)} / {len(ridges)} ridge points")

# --- plot in RA/Dec degrees just for visualization ---
ridges_ra_deg = np.degrees(ridges[:, 1])
ridges_dec_deg = np.degrees(ridges[:, 0])
ridges_clean_ra_deg = np.degrees(ridges_clean[:, 1])
ridges_clean_dec_deg = np.degrees(ridges_clean[:, 0])

plt.figure(figsize=(8,6))
plt.scatter(ridges_ra_deg, ridges_dec_deg, s=5, alpha=0.3, label="All ridges")
plt.scatter(ridges_clean_ra_deg, ridges_clean_dec_deg, s=5, alpha=0.8, label="Kept ridges")
circle = plt.Circle((0,0), mask_radius_deg, color="red", fill=False, linestyle="--", label="Mask hole")
plt.gca().add_patch(circle)
plt.xlabel("RA [deg]")
plt.ylabel("Dec [deg]")
plt.title(f"Ridge filtering test\nradius={radius_arcmin} arcmin, min_coverage={min_coverage}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mock_ridges_filter.png"), dpi=200)
plt.close()

print(f"Saved diagnostic plot to {os.path.join(output_dir,'mock_ridges_filter.png')}")