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
output_dir = "example_zl04_mesh_5e5/shrinked_ridges"
if rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# --- parameters ---
nside = 512
arcmin = np.pi / 180.0 / 60.0
r_bins = np.array([0.0, 2*arcmin, 4*arcmin, 8*arcmin])
min_inner_coverage = 1.0  # stricter choice


def load_mask(mask_filename, nside):
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
    Returns keep mask (bool array).
    """
    vec_ridges = hp.ang2vec(ridge_ra, ridge_dec)  # (N, 3)
    keep_idx = np.zeros(len(ridge_ra), dtype=bool)

    for i, v in enumerate(vec_ridges):
        ok = True
        for j in range(len(r_bins)-1):
            rmin, rmax = r_bins[j], r_bins[j+1]
            annulus_pix = hp.query_annulus(nside, v, rmin, rmax, inclusive=True)
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
    print("[INFO] Loading mask...")
mask = load_mask(mask_filename, nside)
comm.Barrier()

# --- load ridge on rank 0 and broadcast ---
if rank == 0:
    print("Loading ridge HDF5...")
    with h5py.File(ridge_file, "r") as f:
        ridges = f["ridges"][:]  # shape (N, 2) = (dec, ra)
        initial_density = f["initial_density"][:]
        final_density = f["final_density"][:]
else:
    ridges = None
    initial_density = None
    final_density = None

ridges = comm.bcast(ridges, root=0)
initial_density = comm.bcast(initial_density, root=0)
final_density = comm.bcast(final_density, root=0)

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
    ridges_clean = ridges[all_keep]
    out_file = os.path.join(output_dir, "ridges_p15_shrinked.h5")
    with h5py.File(out_file, "w") as f:
        f.create_dataset("ridges", data=ridges_clean)
        f.create_dataset("initial_density", data=initial_density[all_keep])
        f.create_dataset("final_density", data=final_density[all_keep])

    print(f"Saved cleaned ridges to {out_file}")



if rank == 0:
    
    plt.figure(figsize=(8,6))
    plt.scatter(ridges[:,1], ridges[:,0], s=1, alpha=0.3, label="All ridges")
    plt.scatter(ridges_clean[:,1], ridges_clean[:,0], s=1, alpha=0.6, label="Filtered ridges")
    plt.xlabel("RA [rad]")
    plt.ylabel("Dec [rad]")
    plt.title("Ridges before/after edge filtering")
    plt.legend()
    plt.tight_layout()

    plot_file = os.path.join(output_dir, "ridges_filter_diagnostic.png")
    plt.savefig(plot_file, dpi=200)
    plt.close()

    print(f"Saved diagnostic plot to {plot_file}")
