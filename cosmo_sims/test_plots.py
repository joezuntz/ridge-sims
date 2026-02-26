import os


import os, sys

# Directory of this script (cosmo_sims)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up 
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# find modules in the parent directory
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# we stay inside cosmo_sims for outputs
os.chdir(current_dir)



import h5py
import numpy as np
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
    
    
#ra_bg = (ra_bg + 180) % 360    
#ra_bg  = np.radians(ra_bg)
#dec_bg = np.radians(dec_bg)
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

## Contraction test 


#import os, sys
#import numpy as np
#import h5py
#import healpy as hp
#import matplotlib.pyplot as plt

## ----------------- PATH SETUP -----------------
#current_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))
#if parent_dir not in sys.path:
#    sys.path.insert(0, parent_dir)
#os.chdir(current_dir)

## ----------------- INPUTS -----------------
#mask_filename = os.path.join(parent_dir, "des-data", "desy3_gold_mask.npy")

#ridge_file = os.path.join(
#    current_dir,
#    "Cosmo_sim2_ridges/S8/run_3/band_0.1/Ridges_final_p15/S8_run_3_ridges_p15.h5"
#)

## outputs next to the ridge file
#ridge_dir = os.path.dirname(ridge_file)
#out_h5    = os.path.join(ridge_dir, "S8_run_3_ridges_p15_contracted_TEST.h5")
#out_png   = os.path.join(ridge_dir, "S8_run_3_ridges_p15_contracted_TEST.png")

## contraction params
#nside = 1024
#radius_arcmin = 4.0
#min_coverage = 0.1


## ----------------- FUNCTIONS -----------------
#def infer_nside_from_maxpix(maxpix: int):
#    for ns in [256, 512, 1024, 2048, 4096, 8192]:
#        if maxpix < hp.nside2npix(ns):
#            return ns
#    return None


#def load_mask(mask_filename, nside_out, binarize=True):
#    """
#    Loads DES mask from .npy.
#    Handles BOTH cases:
#      (A) file is a full HEALPix map array (length = 12*nside^2)
#      (B) file is a list of hit pixel indices
#    Returns mask at nside_out (RING), optionally binarized.
#    """
#    arr = np.load(mask_filename)

#    # Case A: full map stored
#    if arr.ndim == 1 and arr.size in {hp.nside2npix(256), hp.nside2npix(512), hp.nside2npix(1024),
#                                      hp.nside2npix(2048), hp.nside2npix(4096), hp.nside2npix(8192)}:
#        nside_native = hp.npix2nside(arr.size)
#        mask_native = arr.astype(np.float32)

#    # Case B: hit pixel list stored
#    else:
#        hit_pix = arr.astype(np.int64)
#        if hit_pix.ndim != 1:
#            raise ValueError(f"Unexpected mask array shape {arr.shape}; expected 1D map or 1D hit_pix list.")

#        nside_native = infer_nside_from_maxpix(int(hit_pix.max()))
#        if nside_native is None:
#            raise ValueError("Cannot infer NSIDE from hit_pix.max(); hit_pix may not be HEALPix indices.")

#        mask_native = np.zeros(hp.nside2npix(nside_native), dtype=np.float32)
#        mask_native[hit_pix] = 1.0

#    # Degrade/upgrade to working nside
#    if nside_out != nside_native:
#        mask = hp.ud_grade(mask_native, nside_out=nside_out, power=0)
#    else:
#        mask = mask_native

#    if binarize:
#        mask = (mask > 0.5).astype(np.float32)

#    return mask, nside_native


#def ridge_edge_filter_disk(ridge_ra, ridge_dec, mask, nside, radius_arcmin, min_coverage=1.0):
#    """
#    ridge_ra, ridge_dec are RADIANS.
#    mask is a HEALPix map at NSIDE=nside (RING).
#    """
#    radius = np.radians(radius_arcmin / 60.0)

#    theta_ridges = (np.pi / 2.0) - ridge_dec  # colatitude
#    phi_ridges   = ridge_ra                   # longitude
#    vec_ridges   = hp.ang2vec(theta_ridges, phi_ridges)

#    keep_idx = np.zeros(len(ridge_ra), dtype=bool)
#    fracs = np.empty(len(ridge_ra), dtype=float)
#    ndisk = np.empty(len(ridge_ra), dtype=int)

#    for i, v in enumerate(vec_ridges):
#        disk_pix = hp.query_disc(nside, v, radius, inclusive=True)
#        nd = len(disk_pix)
#        ndisk[i] = nd
#        if nd == 0:
#            frac = 0.0
#        else:
#            frac = mask[disk_pix].sum() / nd
#        fracs[i] = frac
#        keep_idx[i] = (frac >= min_coverage)

#    return keep_idx, fracs, ndisk


## ----------------- RUN -----------------
#if not os.path.isfile(ridge_file):
#    raise FileNotFoundError(ridge_file)

#with h5py.File(ridge_file, "r") as f:
#    ridges = f["ridges"][:]

#ridge_dec = ridges[:, 0]   # radians
#ridge_ra  = ridges[:, 1]   # radians

#mask, mask_native_nside = load_mask(mask_filename, nside_out=nside, binarize=True)
#print("mask native nside:", mask_native_nside, "-> working nside:", nside)
#print("mask stats (min/max/mean):", float(mask.min()), float(mask.max()), float(mask.mean()))

#keep_idx, fracs, ndisk = ridge_edge_filter_disk(
#    ridge_ra, ridge_dec, mask, nside,
#    radius_arcmin=radius_arcmin, min_coverage=min_coverage
#)

#ridges_clean = ridges[keep_idx]

#print("ridge file:", os.path.basename(ridge_file))
#print(f"kept: {len(ridges_clean)}/{len(ridges)}  (fraction={len(ridges_clean)/len(ridges):.6f})")
#print("frac stats (min/median/max):",
#      float(fracs.min()), float(np.median(fracs)), float(fracs.max()))
#print("ndisk stats (min/median/max):",
#      int(ndisk.min()), int(np.median(ndisk)), int(ndisk.max()))

## save contracted
#with h5py.File(out_h5, "w") as f:
#    f.create_dataset("ridges", data=ridges_clean)
#print("saved:", out_h5)

## plot overlay (in degrees)
#plt.figure(figsize=(8, 6))
#plt.scatter(np.degrees(ridge_ra), np.degrees(ridge_dec), s=1, alpha=0.25, label="All ridges")
#if len(ridges_clean) > 0:
#    plt.scatter(np.degrees(ridges_clean[:, 1]), np.degrees(ridges_clean[:, 0]),
#                s=1, alpha=0.8, label="Kept (contracted)")
#plt.xlabel("RA [deg]")
#plt.ylabel("Dec [deg]")
#plt.title(f"Contraction test (radius={radius_arcmin} arcmin, min_cov={min_coverage})")
#plt.legend()
#plt.tight_layout()
#plt.savefig(out_png, dpi=200)
#plt.close()
#print("saved:", out_png)

###########################################################################"

## ===============================
## STATISTICAL MASK–RIDGE TEST
## ===============================

## --- infer native mask NSIDE ---
#hit_pix = np.load(mask_filename)
#maxpix = int(hit_pix.max())

#mask_nside = None
#for ns in [256, 512, 1024, 2048, 4096, 8192]:
#    if maxpix < hp.nside2npix(ns):
#        mask_nside = ns
#        break

#print("mask_nside (inferred):", mask_nside)
#if mask_nside is None:
#    raise ValueError("Cannot infer NSIDE from hit_pix")

## --- build native-resolution binary mask ---
#mask_native = np.zeros(hp.nside2npix(mask_nside), dtype=np.float32)
#mask_native[hit_pix] = 1.0

## --- convert ridge coords to native mask pixels ---
#theta_r = (np.pi/2) - ridge_dec   # ridge_dec already radians
#phi_r   = ridge_ra

#ridge_pix_native = hp.ang2pix(mask_nside, theta_r, phi_r)

## --- 1) How many ridge centers fall inside footprint? ---
#inside_center = mask_native[ridge_pix_native] > 0.5
#frac_inside_center = np.mean(inside_center)

#print("Fraction of ridge centers inside mask:",
#      f"{frac_inside_center:.4f}",
#      f"({inside_center.sum()} / {len(ridge_pix_native)})")

## --- 2) Estimate distance to boundary ---
## A point is "near boundary" if at least one neighbour pixel is outside mask.

#neighbour_outside = np.zeros(len(ridge_pix_native), dtype=bool)

#for i, pix in enumerate(ridge_pix_native[:50000]):  # limit for speed
#    neigh = hp.get_all_neighbours(mask_nside, pix)
#    neigh = neigh[neigh >= 0]
#    if np.any(mask_native[neigh] == 0):
#        neighbour_outside[i] = True

#frac_near_boundary = np.mean(neighbour_outside[:50000])

#print("Approx fraction of ridges near mask boundary (~1 pixel scale):",
#      f"{frac_near_boundary:.4f}")

## --- 3) Mask coverage over ridge RA/Dec bounding box ---
#ra_deg  = np.degrees(ridge_ra)
#dec_deg = np.degrees(ridge_dec)

#print("Ridge RA range (deg):", ra_deg.min(), ra_deg.max())
#print("Ridge Dec range (deg):", dec_deg.min(), dec_deg.max())

#print("Mask global mean coverage:", mask_native.mean())
########################################################################################

import numpy as np
import healpy as hp

mask_filename = os.path.join(parent_dir, "des-data", "desy3_gold_mask.npy")
ridge_file = os.path.join(
    current_dir,
    "Cosmo_sim2_ridges/S8/run_3/band_0.1/Ridges_final_p15/S8_run_3_ridges_p15.h5"
)
arr = np.load(mask_filename)

print("\n===== RAW FILE INFO =====")
print("shape:", arr.shape)
print("dtype:", arr.dtype)
print("ndim:", arr.ndim)
print("min:", arr.min())
print("max:", arr.max())


print("\n===== FULL MAP TEST =====")
if arr.ndim == 1:
    try:
        nside_full = hp.npix2nside(arr.size)
        print("Interpretable as full HEALPix map.")
        print("npix =", arr.size)
        print("nside =", nside_full)
        print("unique values (sample):", np.unique(arr[:1000]))
    except Exception:
        print("Not a full HEALPix map (size not equal to 12*nside^2).")
else:
    print("Not 1D → cannot be full HEALPix map.")

print("\n===== HIT-PIXEL LIST TEST =====")
if arr.ndim == 1:
    maxpix = int(arr.max())
    inferred_nside = None
    for ns in [256, 512, 1024, 2048, 4096, 8192]:
        if maxpix < hp.nside2npix(ns):
            inferred_nside = ns
            break

    if inferred_nside is not None:
        print("Looks like possible hit-pixel list.")
        print("maxpix =", maxpix)
        print("inferred nside =", inferred_nside)
        print("npix at that nside =", hp.nside2npix(inferred_nside))
        print("fraction of sky covered (approx) =", len(arr) / hp.nside2npix(inferred_nside))
    else:
        print("Cannot infer valid nside from maxpix.")
else:
    print("Not 1D → unlikely to be hit-pixel list.")


print("\n===== RING STRUCTURE=====")
if arr.ndim == 1:
    if 'inferred_nside' in locals() and inferred_nside is not None:
        # build ring mask assuming RING
        mask_ring = np.zeros(hp.nside2npix(inferred_nside), dtype=float)
        if arr.size < mask_ring.size:
            mask_ring[arr.astype(int)] = 1.0
            # check distribution over theta rings
            theta, _ = hp.pix2ang(inferred_nside, np.where(mask_ring > 0)[0])
            print("theta (rad) range:", theta.min(), theta.max())
            print("mean theta:", theta.mean())
        else:
            print("arr size equals full map; skipping ring test.")
    else:
        print("No valid inferred_nside; skipping ring test.")




# load ridges (radians)
with h5py.File(ridge_file, "r") as f:
    ridges = f["ridges"][:]
ridge_dec = ridges[:,0]
ridge_ra  = ridges[:,1]
theta = (np.pi/2) - ridge_dec
phi   = ridge_ra

hit_pix = np.load(mask_filename).astype(np.int64)
nside = 4096
npix  = hp.nside2npix(nside)

# Case 1: hit_pix are RING
mask_ring = np.zeros(npix, dtype=np.uint8)
mask_ring[hit_pix] = 1
pix_r = hp.ang2pix(nside, theta, phi)  # assumes RING indexing for pix
inside_ring = mask_ring[pix_r].mean()

# Case 2: hit_pix are NEST -> convert to RING mask
mask_nest = np.zeros(npix, dtype=np.uint8)
mask_nest[hit_pix] = 1
mask_nest_to_ring = hp.reorder(mask_nest, n2r=True)
inside_nest = mask_nest_to_ring[pix_r].mean()

print("Inside fraction assuming hit_pix is RING:", float(inside_ring))
print("Inside fraction assuming hit_pix is NEST:", float(inside_nest))


hit_pix = np.load(mask_filename).astype(np.int64)
nside = 4096

theta, phi = hp.pix2ang(nside, hit_pix)   # RING
ra  = np.degrees(phi)
dec = np.degrees(0.5*np.pi - theta)

print("Mask RA min/max:", ra.min(), ra.max())
print("Mask Dec min/max:", dec.min(), dec.max())
print("Mask RA percentiles (1,5,50,95,99):", np.percentile(ra, [1,5,50,95,99]))
print("Mask Dec percentiles (1,5,50,95,99):", np.percentile(dec, [1,5,50,95,99]))