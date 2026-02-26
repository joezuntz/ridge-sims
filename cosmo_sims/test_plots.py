#import os


#import os, sys

## Directory of this script (cosmo_sims)
#current_dir = os.path.dirname(os.path.abspath(__file__))

## Go one level up 
#parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

## find modules in the parent directory
#if parent_dir not in sys.path:
#    sys.path.insert(0, parent_dir)

## we stay inside cosmo_sims for outputs
#os.chdir(current_dir)



#import h5py
#import numpy as np
#import matplotlib.pyplot as plt

## ---- FILES ----
#bg_file = os.path.abspath(
#    os.path.join(parent_dir, "lhc_cosmo_sims2_zero_err/S8/run_3/source_catalog_0.npy")
#)

#ridge_file = os.path.abspath(
#    os.path.join(current_dir, "Cosmo_sim2_ridges/S8/run_3/band_0.1/Ridges_final_p15/S8_run_3_ridges_p15.h5")
#)
#output_path = "Cosmo_sim2_ridges/overlay_test_S8_run3.png"

## ---- LOAD BACKGROUND ----
#with h5py.File(bg_file, "r") as f:
#    print("BG keys:", list(f.keys()))
#    ra_bg  = f["RA"][:]
#    dec_bg = f["DEC"][:]

## ---- LOAD RIDGES ----
#with h5py.File(ridge_file, "r") as f:
#    ridges = f["ridges"][:]

#dec_ridge = ridges[:, 0]
#ra_ridge  = ridges[:, 1]

## ---- PRINT RANGES (diagnostic) ----
#print("Background RA range:", ra_bg.min(), ra_bg.max())
#print("Background Dec range:", dec_bg.min(), dec_bg.max())
#print("Ridge RA range:", ra_ridge.min(), ra_ridge.max())
#print("Ridge Dec range:", dec_ridge.min(), dec_ridge.max())

## ---- PLOT ----
#plt.figure(figsize=(8,6))

#plt.scatter(ra_bg, dec_bg, s=1, alpha=0.05, label="Background")
#plt.scatter(ra_ridge, dec_ridge, s=1, alpha=0.6, label="Ridges")

#plt.xlabel("RA")
#plt.ylabel("Dec")
#plt.legend()
#plt.tight_layout()

#os.makedirs("Cosmo_sim2_ridges", exist_ok=True)
#plt.savefig(output_path, dpi=200)
#plt.close()

#print(f"Saved overlay plot → {output_path}")




####################################################################################
####################################################################################

# Contraction test 

import os, sys
import numpy as np
import h5py
import healpy as hp
import matplotlib.pyplot as plt

# ----------------- PATH SETUP  -----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
os.chdir(current_dir)

# ----------------- INPUTS : Only on one file  -----------------
mask_filename = os.path.join(parent_dir, "des-data", "desy3_gold_mask.npy")

ridge_file = os.path.join(
    current_dir,
    "Cosmo_sim2_ridges/S8/run_3/band_0.1/Ridges_final_p15/S8_run_3_ridges_p15.h5"
)

# outputs next to the ridge file
ridge_dir = os.path.dirname(ridge_file)
out_h5    = os.path.join(ridge_dir, "S8_run_3_ridges_p15_contracted_TEST.h5")
out_png   = os.path.join(ridge_dir, "S8_run_3_ridges_p15_contracted_TEST.png")

# contraction params
nside = 512
radius_arcmin = 4.0
min_coverage = 0.7


# ----------------- FUNCTIONS -----------------
def load_mask(mask_filename, nside):
    input_mask_nside = 4096
    hit_pix = np.load(mask_filename)
    mask = np.zeros(hp.nside2npix(input_mask_nside))
    mask[hit_pix] = 1
    ##mask = hp.reorder(mask, n2r=True)
    mask = hp.ud_grade(mask, nside_out=nside)
    mask = (mask > 0.5).astype(np.float32)
    return mask

def ridge_edge_filter_disk(ridge_ra, ridge_dec, mask, nside, radius_arcmin, min_coverage=1.0):
    radius = np.radians(radius_arcmin / 60.0)
    theta_ridges = (np.pi / 2.0) - ridge_dec
    phi_ridges   = ridge_ra
    vec_ridges   = hp.ang2vec(theta_ridges, phi_ridges)

    keep_idx = np.zeros(len(ridge_ra), dtype=bool)

    # debug stats
    fracs = np.empty(len(ridge_ra), dtype=float)
    ndisk = np.empty(len(ridge_ra), dtype=int)

    for i, v in enumerate(vec_ridges):
        disk_pix = hp.query_disc(nside, v, radius, inclusive=True)
        ndisk[i] = len(disk_pix)
        if ndisk[i] == 0:
            frac = 0.0
        else:
            frac = mask[disk_pix].sum() / ndisk[i]
        fracs[i] = frac
        keep_idx[i] = (frac >= min_coverage)

    return keep_idx, fracs, ndisk


# ----------------- RUN  -----------------
if not os.path.isfile(ridge_file):
    raise FileNotFoundError(ridge_file)

mask = load_mask(mask_filename, nside)
print("mask stats (min/max/mean):", float(mask.min()), float(mask.max()), float(mask.mean()))

with h5py.File(ridge_file, "r") as f:
    ridges = f["ridges"][:]

ridge_dec = ridges[:, 0]   # radians
ridge_ra  = ridges[:, 1]   # radians

keep_idx, fracs, ndisk = ridge_edge_filter_disk(
    ridge_ra, ridge_dec, mask, nside,
    radius_arcmin=radius_arcmin, min_coverage=min_coverage
)

ridges_clean = ridges[keep_idx]

print(f"ridge file: {os.path.basename(ridge_file)}")
print(f"kept: {len(ridges_clean)}/{len(ridges)}")
print("frac stats (min/median/max):",
      float(np.min(fracs)), float(np.median(fracs)), float(np.max(fracs)))
print("ndisk stats (min/median/max):",
      int(np.min(ndisk)), int(np.median(ndisk)), int(np.max(ndisk)))

# save contracted
with h5py.File(out_h5, "w") as f:
    f.create_dataset("ridges", data=ridges_clean)
print("saved:", out_h5)

# plot overlay (in degrees)
plt.figure(figsize=(8, 6))
plt.scatter(np.degrees(ridge_ra), np.degrees(ridge_dec), s=1, alpha=0.25, label="All ridges")
if len(ridges_clean) > 0:
    plt.scatter(np.degrees(ridges_clean[:, 1]), np.degrees(ridges_clean[:, 0]),
                s=1, alpha=0.8, label="Kept (contracted)")
plt.xlabel("RA [deg]")
plt.ylabel("Dec [deg]")
plt.title(f"Contraction test (radius={radius_arcmin} arcmin, min_cov={min_coverage})")
plt.legend()
plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.close()
print("saved:", out_png)







