import os
import numpy as np
import h5py
import healpy as hp
from mpi4py import MPI
import matplotlib.pyplot as plt

# --- MPI setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- config ---
mask_filename = "des-data/desy3_gold_mask.npy"
ridge_file = "example_zl04_mesh5e5/Ridges_final_p15/ridges_p15.h5"
output_dir = "example_zl04_mesh5e5/shrinked_ridges"

if rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# --- parameters ---
nside = 512
# The r_bins are in radians, but they are defined using a conversion from arcminutes
# The `arcmin` variable converts arcminutes to radians.
arcmin = np.pi / 180.0 / 60.0
r_bins = np.array([0.0, 2*arcmin, 4*arcmin, 8*arcmin])
min_inner_coverage = 1.0  # stricter choice

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

def ridge_edge_filter(ridge_ra, ridge_dec, mask, nside, r_bins, min_inner_coverage):
    """
    Apply ridge-edge filter to a chunk of ridge points.
    
    This function checks the coverage of the input mask around each ridge point
    in different radial bins to filter out ridges on the survey edge. It uses 
    `hp.query_disc` to replicate `hp.query_annulus` for compatibility.
    
    Args:
        ridge_ra (np.ndarray): Array of ridge RA values in degrees.
        ridge_dec (np.ndarray): Array of ridge Dec values in degrees.
        mask (np.ndarray): The HEALPix mask.
        nside (int): The nside of the HEALPix map.
        r_bins (np.ndarray): Radial bins for the annulus, in radians.
        min_inner_coverage (float): The minimum coverage fraction for the inner annulus.
    
    Returns:
        np.ndarray: A boolean array indicating which ridges to keep.
    """
    # Convert RA and DEC from degrees to healpy's spherical coordinates (theta, phi)
    # This correctly handles the `ValueError: THETA is out of range [0,pi]`
    theta_ridges = np.radians(90.0 - ridge_dec)
    phi_ridges = np.radians(ridge_ra)
    
    vec_ridges = hp.ang2vec(theta_ridges, phi_ridges)  # (N, 3)    
    keep_idx = np.zeros(len(ridge_ra), dtype=bool)

    for i, v in enumerate(vec_ridges):
        ok = True
        for j in range(len(r_bins)-1):
            rmin, rmax = r_bins[j], r_bins[j+1]

            # Use hp.query_disc and np.setdiff1d to get the annulus pixels
            # This is the alternative to hp.query_annulus which might not
            # exist in older healpy versions.
            outer_disk_pixels = hp.query_disc(nside, v, rmax, inclusive=True)
            inner_disk_pixels = hp.query_disc(nside, v, rmin, inclusive=True)
            annulus_pix = np.setdiff1d(outer_disk_pixels, inner_disk_pixels)

            if len(annulus_pix) == 0:
                frac = 0.0
            else:
                frac = mask[annulus_pix].sum() / len(annulus_pix)

            if j == 0 and frac < min_inner_coverage:
                ok = False
                break
        keep_idx[i] = ok

    return keep_idx

# --- load mask once on all ranks ---
if rank == 0:
    print(" Loading mask...")
mask = load_mask(mask_filename, nside)
comm.Barrier()

# --- load ridge on rank 0 and broadcast ---
if rank == 0:
    print("Loading ridge...")
    try:
        with h5py.File(ridge_file, "r") as f:
            # Only load the 'ridges' dataset here
            ridges = f["ridges"][:]  # shape (N, 2) = (dec, ra)
    except FileNotFoundError:
        print(f"Error: The file '{ridge_file}' was not found.")
        ridges = None
else:
    ridges = None

# Broadcast only the ridges data
ridges = comm.bcast(ridges, root=0)

if ridges is None:
    if rank == 0:
        print("Exiting due to file loading error.")
    exit()

ridge_dec = ridges[:, 0]
ridge_ra = ridges[:, 1]
n_total = len(ridges)

# --- split work across MPI ranks ---
counts = [n_total // size + (1 if i < n_total % size else 0) for i in range(size)]
displs = np.cumsum([0] + counts[:-1])
start, stop = displs[rank], displs[rank] + counts[rank]

local_ra = ridge_ra[start:stop]
local_dec = ridge_dec[start:stop]

if rank == 0:
    print(f"Total ridges: {n_total}")
    print(f"Distributing work across {size} ranks")

# --- apply filter locally ---
local_keep = ridge_edge_filter(local_ra, local_dec, mask, nside, r_bins, min_inner_coverage)

# --- gather results ---
all_keep = None
if rank == 0:
    all_keep = np.empty(n_total, dtype=bool)

comm.Gatherv(sendbuf=local_keep,
             recvbuf=(all_keep, counts, displs, MPI.BOOL),
             root=0)

# --- save output only on rank 0 ---
if rank == 0:
    # Use the gathered boolean array to filter the ridges
    ridges_clean = ridges[all_keep]

    # Re-open the original HDF5 file to get the full, original density arrays.
    # This is to attempt fixing the IndexError because it ensures we are applying
    # the mask to an array of the same original length.
    try:
        with h5py.File(ridge_file, "r") as f_orig:
            initial_density_orig = f_orig["initial_density"][:]
            final_density_orig = f_orig["final_density"][:]

        # Filter the original density arrays using the boolean mask
        initial_density_clean = initial_density_orig[all_keep]
        final_density_clean = final_density_orig[all_keep]

        out_file = os.path.join(output_dir, "ridges_p15_shrinked.h5")
        with h5py.File(out_file, "w") as f:
            f.create_dataset("ridges", data=ridges_clean)
            f.create_dataset("initial_density", data=initial_density_clean)
            f.create_dataset("final_density", data=final_density_clean)

        print(f"Saved cleaned ridges to {out_file}")
    
        # Plotting code - convert back to degrees for conventional plotting
        ridges_ra_deg = ridges[:, 1]
        ridges_dec_deg = ridges[:, 0]
        ridges_clean_ra_deg = ridges_clean[:, 1]
        ridges_clean_dec_deg = ridges_clean[:, 0]

        plt.figure(figsize=(8,6))
        plt.scatter(ridges_ra_deg, ridges_dec_deg, s=1, alpha=0.3, label="All ridges")
        plt.scatter(ridges_clean_ra_deg, ridges_clean_dec_deg, s=1, alpha=0.6, label="Filtered ridges")
        plt.xlabel("RA [deg]")
        plt.ylabel("Dec [deg]")
        plt.title("Ridges before/after edge filtering")
        plt.legend()
        plt.tight_layout()

        plot_file = os.path.join(output_dir, "ridges_filter_diagnostic.png")
        plt.savefig(plot_file, dpi=200)
        plt.close()

        print(f"Saved diagnostic plot to {plot_file}")
    except KeyError as e:
        print(f"Error: Missing dataset in HDF5 file: {e}. ")
    except FileNotFoundError:
        print(f"Error: The file '{ridge_file}' was not found.")

