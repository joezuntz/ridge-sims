import os, sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import healpy as hp
from mpi4py import MPI


# ==============================================================
# PATH SETUP
# ==============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

os.chdir(current_dir)

import dredge_scms
from ridge_analysis_tools import *

COMM = MPI.COMM_WORLD
RANK = COMM.rank


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def load_catalog_coordinates(base_dir, shift=True, z_cut=None, fraction=None):
    """
    Load coordinates from DES catalog file.
    """
    filename = os.path.join(base_dir, "des-y3-ridges-foreground-v2.h5")
    with h5py.File(filename, 'r') as f:
        ra = f["ra"][:]
        dec = f["dec"][:]
        z_true = f["z"][:]

    # Apply redshift cut
    if z_cut is not None:
        mask = z_true < z_cut
        ra = ra[mask]
        dec = dec[mask]

    # Optional RA shift
    if shift:
        ra = (ra + 180) % 360

    # Apply fractional selection
    if fraction is not None:
        if not (0 < fraction <= 1):
            raise ValueError("fraction must be between 0 and 1")
        n_keep = int(len(ra) * fraction)
        ra = ra[:n_keep]
        dec = dec[:n_keep]

    # Convert (DEC, RA) to radians
    coordinates = np.column_stack((dec, ra))
    coordinates = np.radians(coordinates)
    return coordinates


def results_plot(density_map, ridges, plot_filename):
    """
    Make a plot of a density map and ridge points on top.
    """
    hp.cartview(density_map, min=0, lonra=[20, 50], latra=[-30, 0])
    hp.graticule()

    ridges = np.degrees(ridges)
    ridges_ra = ridges[:, 1] - 180
    ridges_dec = ridges[:, 0]
    hp.projplot(ridges_ra, ridges_dec, 'r.', markersize=1, lonlat=True)
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    plt.close()


def build_density_map(base_dir, nside, smoothing_degrees=0.5, z_cut=None):
    """
    Make a density maps from the coordinates.
    """
    # The healpy conventions are different and should not have
    # the 180 deg shift applied
    data = load_catalog_coordinates(base_dir, shift=False, z_cut=z_cut)
    dec = np.degrees(data[:, 0])
    ra = np.degrees(data[:, 1])
    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    m = np.zeros(npix, dtype=int)
    np.add.at(m, pix, 1)
    m1 = hp.smoothing(m, fwhm=np.radians(smoothing_degrees), verbose=False)
    return m1


def redo_cuts(ridges, initial_density, final_density, initial_percentile=0, final_percentile=25):
    cut1 = initial_density > np.percentile(initial_density, initial_percentile)
    cut2 = final_density > np.percentile(final_density, final_percentile)
    return ridges[cut1 & cut2]


def run_catalog_filament_pipeline(
    bandwidth,
    base_catalog_dir,
    base_label,
    home_dir,
    N=2,
    z_cut=0.4,
    fraction=None):
    """
    Filament-finding pipeline for a single observational catalog (e.g. DES Y3).
    """
    # ----------------------------------------------------------
    # Parameters (identical to simulation pipeline)
    # ----------------------------------------------------------
    neighbours = 5000
    convergence = 1e-5
    seed = 3482364
    mesh_size = int(N * 5e5)

    # ----------------------------------------------------------
    # Output structure
    # ----------------------------------------------------------
    os.makedirs(home_dir, exist_ok=True)

    checkpoint_dir = os.path.join(home_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    plot_dir = os.path.join(home_dir, "plots_by_final_percentile")
    os.makedirs(plot_dir, exist_ok=True)

    ridge_dir = os.path.join(home_dir, "Ridges_analysis")
    os.makedirs(ridge_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Load catalog
    # ----------------------------------------------------------
    coordinates = None
    if RANK == 0:
        coordinates = load_catalog_coordinates(
            base_catalog_dir,
            z_cut=z_cut,
            fraction=fraction
        )

    coordinates = COMM.bcast(coordinates, root=0)

    # ----------------------------------------------------------
    # Filament finder
    # ----------------------------------------------------------
    ridges, initial_density, final_density = dredge_scms.find_filaments(
        coordinates,
        bandwidth=np.radians(bandwidth),
        convergence=np.radians(convergence),
        distance_metric="haversine",
        n_neighbors=neighbours,
        comm=COMM,
        checkpoint_dir=checkpoint_dir,
        resume=True,
        seed=seed,
        mesh_size=mesh_size
    )

    COMM.barrier()

    # ----------------------------------------------------------
    # Output (rank 0)
    # ----------------------------------------------------------
    if RANK == 0:

        final_percentiles = [15]
        initial_percentile = 0

        density_map = build_density_map(
            base_catalog_dir,
            nside=512,
            z_cut=z_cut
        )

        for fp in final_percentiles:

            ridges_cut = redo_cuts(
                ridges,
                initial_density,
                final_density,
                initial_percentile=initial_percentile,
                final_percentile=fp
            )

            # --- HDF5 ---
            h5_filename = os.path.join(
                ridge_dir,
                f"{base_label}_ridges_p{fp:02d}"
                f"__mesh{N}_band{bandwidth:.2f}.h5"
            )

            with h5py.File(h5_filename, "w") as f:
                f.create_dataset("ridges", data=ridges_cut)
                f.create_dataset("initial_density", data=initial_density)
                f.create_dataset("final_density", data=final_density)

            print(f"[rank 0] Saved ridges → {h5_filename}")

            # --- Plot ---
            plot_path = os.path.join(
                plot_dir,
                f"{base_label}_Ridges_plot_p{fp:02d}"
                f"__mesh{N}_band{bandwidth:.2f}.png"
            )
            results_plot(density_map, ridges_cut, plot_path)


def process_ridge_file_local(ridge_file, mask, nside, radius_arcmin, min_coverage, output_dir, plot_dir,
                             out_file=None):  # NEW
    """Apply the filter to one ridge file."""
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
    print(f"[contracted] {os.path.basename(ridge_file)}: kept {len(ridges_clean)}/{n_total}")

    # Save to output folder
    if out_file is None:  # NEW
        base_name = os.path.basename(ridge_file).replace(".h5", "_contracted.h5")
        out_file = os.path.join(output_dir, base_name)
    else:  # NEW
        out_file = os.path.join(output_dir, os.path.basename(out_file))

    with h5py.File(out_file, "w") as f:
        f.create_dataset("ridges", data=ridges_clean)

    # Plot diagnostic
    plot_file = os.path.join(plot_dir, os.path.basename(out_file).replace(".h5", "_diagnostic.png"))
    plt.figure(figsize=(8, 6))
    plt.scatter(np.degrees(ridge_ra), np.degrees(ridge_dec), s=1, alpha=0.3, label="All ridges")  # NEW
    plt.scatter(np.degrees(ridges_clean[:, 1]), np.degrees(ridges_clean[:, 0]), s=1, alpha=0.6, label="Filtered ridges")  # NEW
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title(f"contracted ridges\nradius={radius_arcmin} arcmin, min_cov={min_coverage}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    plt.close()

    print(f"[plot] Saved diagnostic → {plot_file}")

# ==============================================================
# FIXED PARAMETERS
# ==============================================================
bandwidth = 0.1
N = 2
base_label = "DESY3"

output_base = "DES_ridge_analysis"

# contraction parameters
radius_arcmin = 4.0
min_coverage = 0.9
nside = 512

mask_filename = os.path.join(parent_dir, "des-data", "desy3_gold_mask.npy")
mask = np.load(mask_filename) if RANK == 0 else None
mask = COMM.bcast(mask, root=0)


# ==============================================================
# MAIN
# ==============================================================
def main():

    # ----------------------------------------------------------
    # OUTPUT DIRECTORIES
    # ----------------------------------------------------------
    checkpoints_dir = os.path.join(output_base, "checkpoints")
    plots_dir       = os.path.join(output_base, "plots_by_final_percentile")
    ridges_dir      = os.path.join(output_base, "Ridges_analysis")

    if RANK == 0:
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(ridges_dir, exist_ok=True)

    COMM.barrier()

    # ----------------------------------------------------------
    # STAGE 1 — RIDGE FINDING
    # ----------------------------------------------------------
    ridge_file = os.path.join(
        ridges_dir,
        f"{base_label}_ridges_p15__mesh{N}_band{bandwidth:.2f}.h5"
    )

    cat_dir = os.path.join(parent_dir, "des-data")

    if not os.path.exists(ridge_file):

        run_catalog_filament_pipeline(
            bandwidth=bandwidth,
            base_catalog_dir=cat_dir,
            base_label=base_label,
            home_dir=output_base,
            N=N,
            z_cut=0.4
        )

    # ----------------------------------------------------------
    # STAGE 2 — CONTRACTION
    # ----------------------------------------------------------
    contracted_file = ridge_file.replace(".h5", "_contracted_update.h5")                 #update

    if RANK == 0 and os.path.exists(ridge_file) and not os.path.exists(contracted_file):

        process_ridge_file_local(
            ridge_file,
            mask,
            nside,
            radius_arcmin,
            min_coverage,
            ridges_dir,
            plots_dir,
            out_file= contracted_file  # NEW
        )

    COMM.barrier()

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    if RANK == 0:
        summary_file = os.path.join(output_base, "summary.txt")

        with open(summary_file, "w") as f:
            f.write("DES Y3 Ridge Analysis\n")
            f.write("=====================\n\n")
            f.write(f"Bandwidth        : {bandwidth}\n")
            f.write(f"Mesh size (N)    : {N}\n")
            f.write("Catalog          : DES Y3\n")
            f.write("Stages completed : ridge finding + contraction\n")
            f.write("\nOutputs stored in flat directory structure.\n")


# ==============================================================
# EXECUTION
# ==============================================================
if __name__ == "__main__":
    main()




#Temporary
#base_dir = "DES_ridge_analysis/Ridges_analysis"
#fname = os.path.join(base_dir, "DESY3_ridges_p15__mesh2_band0.10_contracted.h5")
#out_png = fname.replace(".h5", ".png")

#with h5py.File(fname, "r") as f:
#    ridges = f["ridges"][:]

#dec = ridges[:, 0]
#ra  = ridges[:, 1]

#plt.figure(figsize=(8, 6))
#plt.scatter(ra, dec, s=1)
#plt.xlabel("RA (rad)")
#plt.ylabel("Dec (rad)")
#plt.title(os.path.basename(fname))
#plt.tight_layout()
#plt.savefig(out_png, dpi=300)
#plt.close()