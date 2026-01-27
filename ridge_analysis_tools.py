import os
import numpy as np
import pandas as pd
import h5py
import time
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import coo_matrix
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
import dredge_scms

from mpi4py import MPI

COMM_WORLD = MPI.COMM_WORLD

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None



###################################################

#---------- NOISE GENERATION ----------------------

###################################################


def transform_background(background_file, output_hdf5_file, seed):
    """Applies random rotation to shear values and saves the transformed background to an HDF5 file."""
    
    np.random.seed(seed)  # Set the seed 
	
    with h5py.File(background_file, "r") as file:
        
        bg_ra = file["RA"][:]
        bg_dec = file["DEC"][:]
        g1_values = file["G1"][:]
        g2_values = file["G2"][:]
        z_true = file["Z_TRUE"][:]
        weights = file["weight"][:] if "weight" in file else np.ones_like(bg_ra)
        
    # Apply random shear rotation to the entire dataset
    psi = np.random.uniform(0, 2 * np.pi, size=len(bg_ra))
    cos_2psi, sin_2psi = np.cos(2 * psi), np.sin(2 * psi)
    
    g1_transformed = cos_2psi * g1_values - sin_2psi * g2_values
    g2_transformed = sin_2psi * g1_values + cos_2psi * g2_values
    
    # save to the new HDF5 file
    with h5py.File(output_hdf5_file, "w") as out_file:
        out_file.create_dataset("RA", data=bg_ra)
        out_file.create_dataset("DEC", data=bg_dec)
        out_file.create_dataset("G1", data=g1_transformed)
        out_file.create_dataset("G2", data=g2_transformed)
        out_file.create_dataset("Z_TRUE", data=z_true)
        if "weight" in file:
            out_file.create_dataset("weight", data=weights)
            
    print(f"Transformed background saved to {output_hdf5_file}")
    
    
    

def transform_DES_background(background_file, output_hdf5_file, seed):
    """Applies random rotation to shear values and saves the transformed background to an HDF5 file."""
    
    np.random.seed(seed)

    with h5py.File(background_file, "r") as file:
        bg_ra = file["ra"][:]
        bg_dec = file["dec"][:]
        g1_values = file["g1"][:]
        g2_values = file["g2"][:]

        has_weight = "weight" in file
        weights = file["weight"][:] if has_weight else np.ones_like(bg_ra)

    # Random rotation
    psi = np.random.uniform(0, 2 * np.pi, size=len(bg_ra))
    cos_2psi = np.cos(2 * psi)
    sin_2psi = np.sin(2 * psi)

    g1_transformed = cos_2psi * g1_values - sin_2psi * g2_values
    g2_transformed = sin_2psi * g1_values + cos_2psi * g2_values

    # Write output (you can choose case here; be consistent downstream)
    with h5py.File(output_hdf5_file, "w") as out_file:
        out_file.create_dataset("ra", data=bg_ra)
        out_file.create_dataset("dec", data=bg_dec)
        out_file.create_dataset("g1", data=g1_transformed)
        out_file.create_dataset("g2", data=g2_transformed)
        if has_weight:
            out_file.create_dataset("weight", data=weights)

    print(f"Transformed background saved to {output_hdf5_file}")






###############################################################

# ------------------ RIDGE FINDER -----------------------------

##############################################################

# Discover LSST directories

def discover_lsst_runs(sim_root):
    """
    Return list of tuples (lsst_label, run_id, full_run_path)
    for all lsst_*/run_*/ directories containing lens_catalog_0.npy.
    """

    discovered = []

    if not os.path.exists(sim_root):
        raise FileNotFoundError(f"Simulation root does not exist: {sim_root}")

    for lsst_dir in sorted(os.listdir(sim_root)):
        lsst_path = os.path.join(sim_root, lsst_dir)
        if not os.path.isdir(lsst_path):
            continue

        # Expect lsst_X
        if not lsst_dir.startswith("lsst_"):
            continue

        for run_dir in sorted(os.listdir(lsst_path)):
            full_run_path = os.path.join(lsst_path, run_dir)
            if not os.path.isdir(full_run_path):
                continue

            # Expect run_Y
            if not run_dir.startswith("run_"):
                continue

            run_id_str = run_dir.replace("run_", "")
            try:
                run_id = int(run_id_str)
            except ValueError:
                continue

            input_file = os.path.join(full_run_path, "lens_catalog_0.npy")
            if os.path.exists(input_file):
                discovered.append((lsst_dir, run_id, full_run_path))

    return discovered



def load_coordinates(base_sim_dir, run_id, shift=True, z_cut=None, fraction=None):
    """
    Load coordinates from a catalog file.

    Parameters
    ----------
    shift : bool
        If True, shifts RA by +180 degrees (mod 360).
    z_cut : float or None
        If provided, select only galaxies with z_true < z_cut.
    fraction : float or None
        If provided, keep only this fraction of the coordinates.
        Must be between 0 and 1.
    """
    
    filename = os.path.join(base_sim_dir, f"run_{run_id}", "lens_catalog_0.npy")
    with h5py.File(filename, 'r') as f:
        ra = f["RA"][:]
        dec = f["DEC"][:]
        z_true = f["Z_TRUE"][:]

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
    hp.cartview(density_map, min=0, lonra=[20, 50], latra=[-30, 0],)
    hp.graticule()

    ridges = np.degrees(ridges)
    ridges_ra = ridges[:, 1] - 180
    ridges_dec = ridges[:, 0]
    hp.projplot(ridges_ra, ridges_dec, 'r.', markersize=1, lonlat=True)
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)


def build_density_map(base_sim_dir, run_id, nside, smoothing_degrees=0.5,z_cut=None):
    """
    Make a density maps from the coordinates.
    """
    # The healpy conventions are different and should not have
    # the 180 deg shift applied
    data = load_coordinates(base_sim_dir, run_id, shift=False,z_cut=z_cut)
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




def run_filament_pipeline(bandwidth, base_sim_dir, run_ids, base_label, home_dir, N = 2, z_cut=0.4,fraction=None):
    """
    Run the full filament-finding + plotting for a given bandwidth, simulation base, and run IDs.
    Results are grouped under the same bandwidth + base label directory.
    z_cut=0.4 is default for DES sims. It is to be changed for lsst sims
    """
    # --- Parameters ---
    neighbours = 5000
    convergence = 1e-5
    seed = 3482364
    mesh_size = int(N * 5e5)

    # --- Directory structure ---
    os.makedirs(home_dir, exist_ok=True)
    checkpoint_dir = os.path.join(home_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for run_id in run_ids:
        # --- Load coordinates on rank 0 only ---
        coordinates = None
        if COMM_WORLD.rank == 0:
            coordinates = load_coordinates(base_sim_dir, run_id, z_cut=z_cut,fraction=None )

        # --- Broadcast to all ranks ---
        coordinates = COMM_WORLD.bcast(coordinates, root=0)

        # --- Parallelized filament finder ---
        ridges, initial_density, final_density = dredge_scms.find_filaments(
            coordinates,
            bandwidth=np.radians(bandwidth),
            convergence=np.radians(convergence),
            distance_metric='haversine',
            n_neighbors=neighbours,
            comm=COMM_WORLD,
            checkpoint_dir=checkpoint_dir,
            resume=True,
            seed=seed,
            mesh_size=mesh_size
        )

        # --- Synchronize all ranks before output ---
        COMM_WORLD.barrier()

        # --- Output (rank 0 only) ---
        if COMM_WORLD.rank == 0:
            final_percentiles = [15]
            initial_percentile = 0

            # Build the density map (rank 0 only)
            density_map = build_density_map(base_sim_dir, run_id, 512, z_cut=z_cut)

            plot_dir = os.path.join(home_dir, "plots_by_final_percentile")
            os.makedirs(plot_dir, exist_ok=True)

            for fp in final_percentiles:
                ridges_cut = redo_cuts(
                    ridges, initial_density, final_density,
                    initial_percentile=initial_percentile,
                    final_percentile=fp
                )

                # Save ridges (by run ID)
                out_dir = os.path.join(home_dir, f"Ridges_final_p{fp:02d}")
                os.makedirs(out_dir, exist_ok=True)
                h5_filename = os.path.join(out_dir, f"{base_label}_run_{run_id}_ridges_p{fp:02d}.h5")

                with h5py.File(h5_filename, 'w') as f:
                    f.create_dataset("ridges", data=ridges_cut)
                    f.create_dataset("initial_density", data=initial_density)
                    f.create_dataset("final_density", data=final_density)

                print(f"[rank 0] Saved ridges → {h5_filename}")

                # Plot
                plot_path = os.path.join(plot_dir, f"{base_label}_run_{run_id}_Ridges_plot_p{fp:02d}.png")
                results_plot(density_map, ridges_cut, plot_path)
                print(f"[rank 0] Saved plot: {plot_path}")

        # --- Synchronize before next run_id ---
        COMM_WORLD.barrier()


###########################################################

# -------------- RIDGE CONTRACTION ------------------------

###########################################################
#Shrink ridges near survey boundaries

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
    theta_ridges = (np.pi / 2.0) - ridge_dec          # NEW: ridge_dec is radians
    phi_ridges = ridge_ra                              # NEW: ridge_ra is radians
    #theta_ridges = np.radians(90.0 - ridge_dec)
    #phi_ridges = np.radians(ridge_ra)
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



#def process_ridge_file_local(ridge_file, mask, nside, radius_arcmin, min_coverage, output_dir, plot_dir):
#    """Apply the filter to one ridge file."""
#    with h5py.File(ridge_file, "r") as f:
#        ridges = f["ridges"][:]
#    ridge_dec = ridges[:, 0]
#    ridge_ra = ridges[:, 1]
#    n_total = len(ridges)

#    keep_idx = ridge_edge_filter_disk(
#        ridge_ra, ridge_dec, mask, nside,
#        radius_arcmin=radius_arcmin, min_coverage=min_coverage
#    )
#    ridges_clean = ridges[keep_idx]
#    print(f"[contracted] {os.path.basename(ridge_file)}: kept {len(ridges_clean)}/{n_total}")

#    # Save to output folder
#    base_name = os.path.basename(ridge_file).replace(".h5", "_contracted.h5")
#    out_file = os.path.join(output_dir, base_name)
#    with h5py.File(out_file, "w") as f:
#        f.create_dataset("ridges", data=ridges_clean)

#    # Plot diagnostic
#    plot_file = os.path.join(plot_dir,os.path.basename(out_file).replace(".h5", "_diagnostic.png"))
    
#    ridge_ra_deg = np.degrees(ridge_ra)                # NEW: convert for plotting
#    ridge_dec_deg = np.degrees(ridge_dec)              # NEW: convert for plotting
#    ridges_clean_ra_deg = np.degrees(ridges_clean[:,1])# NEW: convert for plotting
#    ridges_clean_dec_deg = np.degrees(ridges_clean[:,0])# NEW: convert for plotting

#    plt.figure(figsize=(8, 6))
#    plt.scatter(ridge_ra_deg, ridge_dec_deg, s=1, alpha=0.3, label="All ridges")                 # NEW
#    plt.scatter(ridges_clean_ra_deg, ridges_clean_dec_deg, s=1, alpha=0.6, label="Filtered ridges")# NEW
    

#    plt.xlabel("RA [deg]")
#    plt.ylabel("Dec [deg]")
#    plt.title(f"contracted ridges\nradius={radius_arcmin} arcmin, min_cov={min_coverage}")
#    plt.legend()
#    plt.tight_layout()
#    plt.savefig(plot_file, dpi=200)
#    plt.close()

#    print(f"[plot] Saved diagnostic → {plot_file}")

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
    plt.scatter(np.degrees(ridge_ra), np.degrees(ridge_dec), s=1, alpha=0.3, label="All ridges")  # NEW (units)
    plt.scatter(np.degrees(ridges_clean[:, 1]), np.degrees(ridges_clean[:, 0]), s=1, alpha=0.6, label="Filtered ridges")  # NEW (units)
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title(f"contracted ridges\nradius={radius_arcmin} arcmin, min_cov={min_coverage}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    plt.close()

    print(f"[plot] Saved diagnostic → {plot_file}")





##########################################################

#-------------- RIDGE SEGMENTATION --------------------

##########################################################

def build_mst(points, k=10):
    """Constructs a Minimum Spanning Tree (MST) from given points."""
    tree = KDTree(points)
    distances, indices = tree.query(points, k=k+1)
    
    row, col, data = [], [], []
    for i in range(len(points)):
        for j in range(1, k+1):
            row.append(i)
            col.append(indices[i, j])
            data.append(distances[i, j])
    
    sparse_dist_matrix = coo_matrix((data, (row, col)), shape=(len(points), len(points)))
    mst_sparse = minimum_spanning_tree(sparse_dist_matrix).tocoo()

    G = nx.Graph()
    for i, j, weight in zip(mst_sparse.row, mst_sparse.col, mst_sparse.data):
        G.add_edge(int(i), int(j), weight=weight)

    return G

def detect_branch_points(mst):
    """Find nodes with degree > 2 (branch points)."""
    return [node for node, degree in dict(mst.degree()).items() if degree > 2]

def split_mst_at_branches(mst, branch_points):
    """Splits the MST into connected components after removing branch points."""
    G = mst.copy()
    G.remove_nodes_from(branch_points)
    return list(nx.connected_components(G))

def segment_filaments_with_dbscan(points, filament_segments, eps=0.02, min_samples=5):
    """Clusters MST segments using DBSCAN."""
    labels = np.full(len(points), -1)
    cluster_id = 0
    
    for segment in filament_segments:
        segment_points = np.array([points[idx] for idx in segment])
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        segment_labels = dbscan.fit_predict(segment_points)
        
        for i, idx in enumerate(segment):
            if segment_labels[i] != -1:
                labels[idx] = cluster_id + segment_labels[i]  
        
        cluster_id += max(segment_labels) + 1 if len(segment_labels) > 0 else 0
    
    return labels


def save_filaments_to_hdf5(ra_dec, labels, filename, dataset_name="data"):
    """Save RA, DEC, and Filament Labels with RA from column 1 and DEC from column 0"""
    dtype = [("RA", "f8"), ("DEC", "f8"), ("Filament_Label", "i8")]  
    structured_data = np.array(
        [(ra_dec[i, 1], ra_dec[i, 0], label) for i, label in enumerate(labels)],
        dtype=dtype
    )

    with h5py.File(filename, "w") as hdf:
        hdf.create_dataset(dataset_name, data=structured_data)



	
#################################################################

# ------------------- GET COSMO SIM FILES -----------------------

################################################################


def read_sim_background(bg_file, rows, comm=None):
    """Read background galaxies from simulated catalog (HDF5)."""
    with h5py.File(bg_file, "r") as f:
        if rows is None:
            rows = f["RA"].shape[0]
        if comm is None:
            s = slice(None)
        else:
            row_per_process = rows // comm.size
            s = slice(comm.rank * row_per_process, (comm.rank + 1) * row_per_process)
            print("Process rank:", comm.rank, "reading rows:", s, "out of rows:", f["RA"].shape)

        bg_ra = f["RA"][s]
        bg_ra = (bg_ra + 180) % 360
        bg_dec = f["DEC"][s]
        g1 = f["G1"][s]
        g2 = f["G2"][s]
        z_true = f["Z_TRUE"][s]
        weights = f["weight"][s] if "weight" in f else np.ones_like(bg_ra)

    return bg_ra, bg_dec, g1, g2, z_true, weights



#def read_sim_background(bg_file, stride=1000):
#    """
#    Read background galaxies from simulated catalog (HDF5).
#    Loads the full dataset but only keeps every `stride`-th row.
#    """
#    with h5py.File(bg_file, "r") as f:
#        bg_ra = f["RA"][::stride]
#        bg_ra = (bg_ra + 180) % 360  
#        bg_dec = f["DEC"][::stride]
#        g1 = f["G1"][::stride]
#        g2 = f["G2"][::stride]
#        z_true = f["Z_TRUE"][::stride]
#        weights = f["weight"][::stride] if "weight" in f else np.ones_like(bg_ra)

#    return bg_ra, bg_dec, g1, g2, z_true, weights



#def read_DES_background(bg_file, comm=None):
#    """Read background galaxies from DES-like catalog with 'background' group."""
#    with h5py.File(bg_file, "r") as f:
#        total_rows = f["background"]["ra"].shape[0]
#        if comm is None:
#            s = slice(None)
#        else:
#            row_per_process = total_rows // comm.size
#            s = slice(comm.rank * row_per_process, (comm.rank + 1) * row_per_process)

#        bg_ra = f["background"]["ra"][s]
#        bg_ra = (bg_ra + 180) % 360
#        bg_dec = f["background"]["dec"][s]
#        g1 = f["background"]["g1"][s]
#        g2 = f["background"]["g2"][s]
#        weights = f["background"]["weight"][s]

#    return bg_ra, bg_dec, g1, g2, None, weights  # DES has no z_true, return None

def read_DES_background(bg_file, comm=None):
    """
    Read background galaxies from DES Y3 catalog with flat structure:
    ra, dec, g1, g2, weight, z
    """
    with h5py.File(bg_file, "r") as f:

        total_rows = f["ra"].shape[0]

        if comm is None:
            s = slice(None)
        else:
            row_per_process = total_rows // comm.size
            s = slice(
                comm.rank * row_per_process,
                (comm.rank + 1) * row_per_process
            )

        bg_ra = f["ra"][s]
        #bg_ra = (bg_ra + 180) % 360                                    # Commented 

        bg_dec = f["dec"][s]
        g1 = f["g1"][s]
        g2 = f["g2"][s]

        z = f["z"][s]

        weights = (
            f["weight"][s]
            if "weight" in f
            else np.ones_like(bg_ra)
        )

    return bg_ra, bg_dec, g1, g2, z, weights

def read_DES_noise_background(bg_file, comm=None):
    """
    Read DES noise background catalog with flat structure:
    ra, dec, g1, g2, weight
    (no z stored in file)

    Returns:
      bg_ra, bg_dec, g1, g2, z_dummy, weights

    z_dummy is returned as NaNs 
    """
    with h5py.File(bg_file, "r") as f:

        total_rows = f["ra"].shape[0]

        if comm is None:
            s = slice(None)
        else:
            row_per_process = total_rows // comm.size
            s = slice(comm.rank * row_per_process, (comm.rank + 1) * row_per_process)

        bg_ra = f["ra"][s]
        #bg_ra = (bg_ra + 180) % 360

        bg_dec = f["dec"][s]
        g1 = f["g1"][s]
        g2 = f["g2"][s]

        weights = f["weight"][s] if "weight" in f else np.ones_like(bg_ra)

        # Dummy z to preserve the interface expected by downstream code
        z_dummy = np.full_like(bg_ra, np.nan, dtype=float)

    return bg_ra, bg_dec, g1, g2, z_dummy, weights



################ To read use later in the DES shear code withought the shift ##########


def read_DES_background_noshift(bg_file, comm=None):
    """
    Read DES Y3 catalog EXACTLY as stored (no RA shift applied).
    Expected datasets: ra, dec, g1, g2, z, weight
    """
    with h5py.File(bg_file, "r") as f:
        total_rows = f["ra"].shape[0]

        if comm is None:
            s = slice(None)
        else:
            row_per_process = total_rows // comm.size
            s = slice(comm.rank * row_per_process, (comm.rank + 1) * row_per_process)

        bg_ra = f["ra"][s]     # <-- NO SHIFT
        bg_dec = f["dec"][s]
        g1 = f["g1"][s]
        g2 = f["g2"][s]
        z = f["z"][s]
        weights = f["weight"][s] if "weight" in f else np.ones_like(bg_ra)

    return bg_ra, bg_dec, g1, g2, z, weights


def read_DES_noise_background_noshift(bg_file, comm=None):
    """
    Read DES noise catalog EXACTLY as stored (no RA shift applied).
    Expected datasets: ra, dec, g1, g2, weight
    Returns a dummy z (NaNs) to keep the interface consistent.
    """
    with h5py.File(bg_file, "r") as f:
        total_rows = f["ra"].shape[0]

        if comm is None:
            s = slice(None)
        else:
            row_per_process = total_rows // comm.size
            s = slice(comm.rank * row_per_process, (comm.rank + 1) * row_per_process)

        bg_ra = f["ra"][s]     # <-- NO SHIFT
        bg_dec = f["dec"][s]
        g1 = f["g1"][s]
        g2 = f["g2"][s]
        weights = f["weight"][s] if "weight" in f else np.ones_like(bg_ra)

        z_dummy = np.full_like(bg_ra, np.nan, dtype=float)

    return bg_ra, bg_dec, g1, g2, z_dummy, weights
################################################################################


def load_background(bg_file, comm=None, rows=None, background_type=None):
    if background_type == "sim":
        return read_sim_background(bg_file, rows, comm=comm)

    elif background_type == "DES":
        return read_DES_background(bg_file, comm=comm)

    elif background_type == "noise":
        return read_DES_noise_background(bg_file, comm=comm)

    # ---- NEW: use these when files already contain shifted RA ----
    elif background_type == "DES_noshift":
        return read_DES_background_noshift(bg_file, comm=comm)

    elif background_type == "noise_noshift":
        return read_DES_noise_background_noshift(bg_file, comm=comm)

    else:
        raise ValueError(f"Unknown background_type: {background_type}")


def find_contracted_files(home_dir):
    """find all '_contracted.h5' ridge files in directory."""
    contracted_files = []
    for root, _, files in os.walk(home_dir):
        for f in files:
            if f.endswith("_contracted.h5"):
                contracted_files.append(os.path.join(root, f))
    return contracted_files


def find_background_file(h5_file, base_sim_root):
    """
    This is the file structure for all cosmological simulations: 
    Given a ridge file path like:
        Cosmo_sim_ridges/S8/run_1/band_0.1/Ridges_final_p15/..._contracted.h5
    Return the corresponding background catalog path:
        lhc_cosmo_sims_zero_err/S8/run_1/source_catalog_cutzl04.h5
    """
    parts = h5_file.split(os.sep)
    try:
        cat_index = parts.index("Cosmo_sim_ridges") + 1
        category = parts[cat_index]
        run_folder = parts[cat_index + 1]
    except (ValueError, IndexError):
        raise RuntimeError(f"Unexpected ridge file path structure: {h5_file}")

    bg_file = os.path.join(base_sim_root, category, run_folder, "source_catalog_cutzl04.h5")
    if not os.path.exists(bg_file):
        raise FileNotFoundError(f"Background file not found: {bg_file}")
    return bg_file
    
    

############################################################################

# -------------------- BACKGROUND REDSHIFT CUT -----------------------------

############################################################################

# === Locate the .npy background file for a given cosmo ridge file
def find_background_npy(h5_file, base_sim_root):
    """
    This is the file structure for all cosmo background files
    Given a ridge file path like:
        Cosmo_sim_ridges/S8/run_1/band_0.1/Ridges_final_p15/..._contracted.h5
    Return the corresponding .npy background file path:
        lhc_run_sims/S8/run_1/source_catalog_0.npy
    """
    parts = h5_file.split(os.sep)
    try:
        cat_index = parts.index("Cosmo_sim_ridges") + 1
        category = parts[cat_index]
        run_folder = parts[cat_index + 1]
    except (ValueError, IndexError):
        raise RuntimeError(f"Unexpected ridge file path structure: {h5_file}")

    npy_file = os.path.join(base_sim_root, category, run_folder, "source_catalog_0.npy")
    if not os.path.exists(npy_file):
        raise FileNotFoundError(f"Background .npy file not found: {npy_file}")
    return npy_file


# === Convert .npy → filtered .h5 (z>0.4 is applied by default)

def convert_npy_to_filtered_h5(npy_path, z_cut=0.4):
    """
    Load .npy (actually HDF5), apply z>z_cut and finite-value filters,
    and save to .h5 with same structure as cosmological shear input.
    """

    run_dir = os.path.dirname(npy_path)
    output_file_path = os.path.join(
        run_dir, f"source_catalog_cutzl{z_cut:.2f}.h5"
    )

    # load file 
    with h5py.File(npy_path, "r") as hf:
        bg_ra_full     = hf["RA"][:]
        bg_dec_full    = hf["DEC"][:]
        g1_values_full = hf["G1"][:]
        g2_values_full = hf["G2"][:]
        z_true_full    = hf["Z_TRUE"][:]
        weights_full   = hf["weight"][:] if "weight" in hf else np.ones_like(bg_ra_full)

    # Apply z-cut
    z_mask = z_true_full > z_cut
    valid_mask = (
        np.isfinite(bg_ra_full)
        & np.isfinite(bg_dec_full)
        & np.isfinite(g1_values_full)
        & np.isfinite(g2_values_full)
        & np.isfinite(weights_full)
        & z_mask
    )

    # Filter
    bg_ra_filtered     = bg_ra_full[valid_mask]
    bg_dec_filtered    = bg_dec_full[valid_mask]
    g1_values_filtered = g1_values_full[valid_mask]
    g2_values_filtered = g2_values_full[valid_mask]
    z_true_filtered    = z_true_full[valid_mask]
    weights_filtered   = weights_full[valid_mask]

    # Save output h5
    with h5py.File(output_file_path, "w") as hf:
        hf.create_dataset("RA", data=bg_ra_filtered)
        hf.create_dataset("DEC", data=bg_dec_filtered)
        hf.create_dataset("G1", data=g1_values_filtered)
        hf.create_dataset("G2", data=g2_values_filtered)
        hf.create_dataset("Z_TRUE", data=z_true_filtered)
        hf.create_dataset("weight", data=weights_filtered)

    print(f"Filtered background data saved → {output_file_path}")
    print(f"Original: {len(bg_ra_full)} | Filtered: {len(bg_ra_filtered)}")


def convert_all_backgrounds(base_sim_root, z_cut=0.4):   # updated
    """
    convert all .npy backgrounds.
    """

    SKIP_IF_EXISTS = True  
    REPORT_AT_END = True   

    skipped_existing = []
    missing_input = []

    if not os.path.exists(base_sim_root):
        raise FileNotFoundError(f"Base directory not found: {base_sim_root}")

    print(f"[INFO] Scanning simulation tree: {base_sim_root}")

    for root, _, files in os.walk(base_sim_root):
        if "source_catalog_0.npy" not in files:
            continue

        npy_path = os.path.join(root, "source_catalog_0.npy")

        # FIX: properly format output filename with z_cut
        output_path = os.path.join(
            root, f"source_catalog_cutzl{z_cut:.2f}.h5"   # updated
        )

        # skip if output exists
        if SKIP_IF_EXISTS and os.path.exists(output_path):
            print(f"[SKIP] Output exists → {output_path}")
            skipped_existing.append(npy_path)
            continue

        try:
            convert_npy_to_filtered_h5(npy_path, z_cut=z_cut)   # updated (pass z_cut)
        except FileNotFoundError:
            print(f"[MISSING] Input .npy file not found: {npy_path}")
            missing_input.append(npy_path)

    # END REPORT
    if REPORT_AT_END:
        print("\n========= CONVERSION SUMMARY =========")

        if skipped_existing:
            print(f"[SKIPPED EXISTING] {len(skipped_existing)} files:")
            for p in skipped_existing:
                print(f"  - {p}")

        if missing_input:
            print(f"[MISSING INPUT] {len(missing_input)} files:")
            for p in missing_input:
                print(f"  - {p}")

        if not (skipped_existing or missing_input):
            print("[OK] All backgrounds processed successfully.")
        print("=======================================")




def discover_and_convert_BG(base_root, z_cut=0.4):   # updated
    """
    Find all run dirs and convert with z_cut.
    """

    if not os.path.exists(base_root):
        raise FileNotFoundError(f"Directory not found: {base_root}")

    run_dirs = []

    for lsst_dir in sorted(os.listdir(base_root)):
        full_lsst_path = os.path.join(base_root, lsst_dir)
        if not os.path.isdir(full_lsst_path):
            continue

        for run_dir in sorted(os.listdir(full_lsst_path)):
            full_run_path = os.path.join(full_lsst_path, run_dir)
            if not os.path.isdir(full_run_path):
                continue

            npy_file = os.path.join(full_run_path, "source_catalog_0.npy")
            if os.path.exists(npy_file):
                run_dirs.append(full_run_path)

    print(f"[INFO] Found {len(run_dirs)} run directories with backgrounds")

    for rd in run_dirs:
        print(f"\n=== Processing {rd} ===")
        convert_all_backgrounds(rd, z_cut=z_cut)   # updated



def convert_DES_background_with_zcut(
    bg_file,
    z_cut=0.4,
    comm=None,
):
    """
    Read DES Y3 background catalog (flat structure),
    apply a redshift cut z > z_cut,
    and write a filtered catalog:

      - in the SAME directory as the input file
      - with the SAME structure 
      - with the SAME filename, plus suffix _cutzl{z_cut:.2f} before extension

    Expected input datasets:
      /ra, /dec, /g1, /g2, /z, /weight

    Output datasets (same):
      /ra, /dec, /g1, /g2, /z, /weight
    """

    # --------------------------------------------------
    # input path
    # --------------------------------------------------
    bg_file = os.path.abspath(bg_file)

    if not os.path.exists(bg_file):
        raise FileNotFoundError(f"Background file not found: {bg_file}")

    # --------------------------------------------------
    # Output path: 
    # --------------------------------------------------
    base_dir = os.path.dirname(bg_file)
    base_name = os.path.basename(bg_file)
    name, ext = os.path.splitext(base_name)

    out_file = os.path.join(
        base_dir,
        f"{name}_cutzl{z_cut:.2f}{ext}"
    )

    # --------------------------------------------------
    # Read input 
    # --------------------------------------------------
    ra, dec, g1, g2, z, weight = read_DES_background(bg_file, comm=comm)

    # --------------------------------------------------
    # Build validity mask + z cut
    # --------------------------------------------------
    valid = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(g1)
        & np.isfinite(g2)
        & np.isfinite(z)
        & np.isfinite(weight)
        & (z > z_cut)
    )

    # --------------------------------------------------
    # Write output 
    # --------------------------------------------------
    if comm is not None and comm.rank != 0:
        return out_file

    # Guard against accidental overwrite
    if os.path.exists(out_file):
        raise FileExistsError(
            f"Output file already exists: {out_file}\n"
            "Delete it or choose a different z_cut."
        )

    with h5py.File(out_file, "w") as f:
        f.create_dataset("ra", data=ra[valid])
        f.create_dataset("dec", data=dec[valid])
        f.create_dataset("g1", data=g1[valid])
        f.create_dataset("g2", data=g2[valid])
        f.create_dataset("z", data=z[valid])
        f.create_dataset("weight", data=weight[valid])

        f.attrs["z_cut"] = float(z_cut)
        f.attrs["source"] = os.path.basename(bg_file)

    print("[DONE] DES background z-cut applied")
    print(f"       Input:    {bg_file}")
    print(f"       Output:   {out_file}")
    print(f"       Selected: {int(valid.sum())} / {len(ra)}")

    return out_file




##############################################################################

# ------------------------- RIDGE SHEAR PROCESS -----------------------------------

##############################################################################

def get_position_angle(ra_source, dec_source, ra_filament, dec_filament):
    """
    All inputs are expected to be in radians
    """
    # matched_filament_points and bg_coords are in radians right now.
    # So we need to convert them back to degrees for SkyCoord
    bg_sky = SkyCoord(ra=ra_source * u.rad, dec=dec_source * u.rad)
    filament_sky = SkyCoord(ra=ra_filament * u.rad, dec=dec_filament * u.rad)

    # Compute position angle of filament point relative to background galaxy
    phi = bg_sky.position_angle(filament_sky).rad + np.pi/2  # radians, same as before
    return phi


def precompute_pixel_regions(ras, decs, g1, g2, z, weights, nside_coverage):
    """
    Split up the source catalot into low-resolution healpix pixels for fast lookup later on.

    Parameters
    ----------
    ras : numpy.ndarray
        The RA values of the sources in degrees.
    decs : numpy.ndarray
        The DEC values of the sources in degrees.
    g1 : numpy.ndarray
        The G1 shear values of the sources.
    g2 : numpy.ndarray
        The G2 shear values of the sources.
    nside_coverage : int
        The nside of the low-resolution healpix map to use for coverage.

    Returns
    -------
    pixel_regions : dict
        A dictionary mapping low-resolution healpix pixel indices to tuples of
        (ras, decs, g1, g2) for the sources in that pixel. 
    """
    assert np.max(ras) < 3 * np.pi, "RA values should be in radians"

    # 110 arcmin, so as we are only going out to 1 degree that should be enough for any
    # adjacent pixels.
    source_healpix_low_res = hp.ang2pix(nside_coverage, np.pi/2 - decs, ras)
    n_unique_pixels = len(np.unique(source_healpix_low_res))
    print(f"Precomputing source pixel regions: {n_unique_pixels} unique pixels at nside {nside_coverage}")
    # 1715 pixels, so a nice reduction but not too small.
    pixel_regions = {}

    # This takes < 1 minute
    for j, i in enumerate(np.unique(source_healpix_low_res)):
        index = np.where(source_healpix_low_res == i)[0]
        pixel_regions[i] = (ras[index], decs[index], g1[index], g2[index], z[index], weights[index])

    return pixel_regions



def get_nearby_sources(raf, decf, pixel_regions, nside_coverage):
    """
    Extract from the pixel regions the sources that are near to the filament points.
    """
    filament_pix_low_res = hp.ang2pix(nside_coverage, np.pi/2 - decf, raf)
    filament_pix_low_res = np.unique(filament_pix_low_res)
    pixels_needed = hp.get_all_neighbours(nside_coverage, filament_pix_low_res).flatten()

    # Include the filament pixels themselves, and then get unique values
    pixels_needed = np.unique(np.concatenate((pixels_needed, filament_pix_low_res)))
    ra = []
    dec = []
    g1 = []
    g2 = []
    z = []
    weight = []

    for p in pixels_needed:
        if p in pixel_regions:
            ras_s, decs_s, g1_s, g2_s, z_s, w_s  = pixel_regions[p]
            ra.append(ras_s)
            dec.append(decs_s)
            g1.append(g1_s)
            g2.append(g2_s)
            z.append(z_s)
            weight.append(w_s)

    if len(ra) == 0:
        return None, None, None, None, None, None, None # Add 2xNone because othewise it has problem unpacking all 7 argument

    ra = np.concatenate(ra)
    dec = np.concatenate(dec)
    g1 = np.concatenate(g1)
    g2 = np.concatenate(g2)
    z = np.concatenate(z)
    weight = np.concatenate(weight)
    source_coords = np.array([dec, ra]).T

    return source_coords, ra, dec, g1, g2, z, weight

def process_shear_sims(filament_file, bg_data, output_shear_file, k=1, num_bins=20, comm=comm,
                       flip_g1=False, flip_g2=False, background_type=None, nside_coverage=32,
                       min_distance_arcmin=1.0, max_distance_arcmin=60.0):

    min_ang_rad = np.radians(min_distance_arcmin/60)
    max_ang_rad = np.radians(max_distance_arcmin/60)
    
    # this line was defined lower in the code
    #bins = np.logspace(np.log10(min_ang_rad), np.log10(max_ang_rad), num_bins + 1)
    
    coverage_pixel_size = hp.nside2resol(nside_coverage, arcmin=True)
    assert coverage_pixel_size > max_distance_arcmin, "Coverage pixel size ({coverage_pixel_size} arcmin) must be larger than max_distance_arcmin ({max_distance_arcmin} arcmin). Increase nside_coverage."

    start_time = time.time()


    with h5py.File(filament_file, "r") as hdf:
        dataset = hdf["data"]
        ra_filaments = dataset["RA"][:]
        dec_filaments = dataset["DEC"][:]
        labels = dataset["Filament_Label"][:]

    ra_background, dec_background, g1_background, g2_background, z_background, weights_background = load_background(
        bg_data, comm=comm, background_type=background_type)
    ra_background = np.radians(ra_background)
    dec_background = np.radians(dec_background)
    
    if flip_g1:
        g1_background *= -1
    if flip_g2:
        g2_background *= -1

    bin_sums_plus = np.zeros(num_bins)
    bin_sums_cross = np.zeros(num_bins)
    bin_weighted_distances = np.zeros(num_bins)
    bin_weights = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    rank = comm.rank if comm is not None else 0

    # from sklearn docs:
    #  Note that the haversine distance metric requires data in the form of [latitude, longitude] 
    # and both inputs and outputs are in units of radians.
    # check units are in degrees
    assert ra_filaments.max() < 3 * np.pi, "Filament RA values should be in radians {}".format(ra_filaments.max())
    assert ra_background.max() < 3 * np.pi, "Background RA values should be in radians {}".format(ra_background.max())


    # Pre-split the catalog into a low-resolution map, so that we can just look up pixels
    # that are relatively close to the filament points later.
    pixel_regions = precompute_pixel_regions(
        ra_background, dec_background,
        g1_background, g2_background,
        z_background, weights_background,
        nside_coverage
    )
    

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]  # Exclude noise label (-1)

    for filament_index, label in enumerate(unique_labels):
        filament_mask = np.where(labels == label)[0]
        filament_coords = np.array([dec_filaments[filament_mask], ra_filaments[filament_mask]]).T

        # Pull out sources within adjacent low-resolution healpix pixels
        source_coords, ra_subset, dec_subset, g1_subset, g2_subset, z_subset, weights_subset = get_nearby_sources(
            ra_filaments[filament_mask], dec_filaments[filament_mask], pixel_regions, nside_coverage
        )

        # There may be no sources near this filament segment
        if source_coords is None:
            continue


        if (rank == 0) and (filament_index % 10 == 0):
            print(f"[{rank}] Processing filament {filament_index} / {len(unique_labels)} - {source_coords.shape[0]} nearby sources")


        # For each background galaxy, find nearest filament point
        nbrs = NearestNeighbors(n_neighbors=1, leaf_size=100, metric="haversine").fit(filament_coords)
        distances, indices = nbrs.kneighbors(source_coords)
        matched_filament_points = filament_coords[indices[:, 0]]


        # Get the rotation angle phi between the background galaxy and the filament point
        phi = get_position_angle(
            ra_source=ra_subset,
            dec_source=dec_subset,
            ra_filament=matched_filament_points[:, 1],
            dec_filament=matched_filament_points[:, 0],
        )
        
        # Rotate the shear into the filament frame
        g_plus = -g1_subset * np.cos(2 * phi) + g2_subset * np.sin(2 * phi)
        g_cross = g1_subset * np.sin(2 * phi) - g2_subset * np.cos(2 * phi)

        # Bin the distances between 1 arcmin and 1 degree
        bins = np.logspace(np.log10(min_ang_rad), np.log10(max_ang_rad), num_bins + 1) # This is now moved to the top
        bin_indices = np.digitize(distances[:, 0], bins) - 1
        valid_bins = (bin_indices >= 0) & (bin_indices < num_bins)

        # Accumulate the total tangential and cross shear in each bin,
        # together with the counts, weights, and actual (as opposed to nominal) distances.
        np.add.at(bin_sums_plus, bin_indices[valid_bins], weights_subset[valid_bins] * g_plus[valid_bins])
        np.add.at(bin_sums_cross, bin_indices[valid_bins], weights_subset[valid_bins] * g_cross[valid_bins])
        np.add.at(bin_weighted_distances, bin_indices[valid_bins], weights_subset[valid_bins] * distances[valid_bins, 0])
        np.add.at(bin_weights, bin_indices[valid_bins], weights_subset[valid_bins])
        np.add.at(bin_counts, bin_indices[valid_bins], 1)
    
    # Sum up the results from all processes, so the totals are correct
    sum_in_place(bin_sums_plus, comm)
    sum_in_place(bin_sums_cross, comm)
    sum_in_place(bin_weighted_distances, comm)
    sum_in_place(bin_weights, comm)
    sum_in_place(bin_counts, comm)

    # Only the root process computes the final division and writes out the final results
    if comm is not None and comm.rank != 0:
        return
        
    # Avoid NaNs by only dividing where bin_weights > 0
    weighted_g_plus = np.divide(bin_sums_plus, bin_weights, out=np.zeros_like(bin_sums_plus), where=bin_weights > 0)
    weighted_g_cross = np.divide(bin_sums_cross, bin_weights, out=np.zeros_like(bin_sums_cross), where=bin_weights > 0)
    weighted_real_distances = np.divide(bin_weighted_distances, bin_weights, out=np.zeros_like(bin_weighted_distances), where=bin_weights > 0)

    # Get the nominal bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Save the output data to CSV format
    output_data = np.column_stack((bin_centers, weighted_real_distances, weighted_g_plus, weighted_g_cross, bin_counts, bin_weights))
    np.savetxt(output_shear_file, output_data, delimiter=",", header="Bin_Center,Weighted_Real_Distance,Weighted_g_plus,Weighted_g_cross,Counts,bin_weight", comments="")

    print(f"Shear processing completed in {time.time() - start_time:.2f} seconds.")
        



def process_ridge_file(h5_file, BG_data, filament_h5, shear_csv, background_type, shear_flip_csv = None, comm=None):
    """
    Compute MST → filaments → shear from a contracted ridge file.
    All paths are passed explicitly to keep the function file-agnostic.
    """
    if comm is None or comm.rank == 0:
        print(f"[rank 0] Processing {h5_file}")

        with h5py.File(h5_file, 'r') as f:
            Ridges = f["ridges"][:]

        mst = build_mst(Ridges)
        branch_points = detect_branch_points(mst)
        filament_segments = split_mst_at_branches(mst, branch_points)
        filament_labels = segment_filaments_with_dbscan(Ridges, filament_segments)

        save_filaments_to_hdf5(Ridges, filament_labels, filament_h5)
        print(f"[save] Filaments → {filament_h5}")

    if comm is not None:
        comm.Barrier()

    # --- Shear calculations ---
    process_shear_sims(
        filament_h5, BG_data, output_shear_file=shear_csv,
        k=1, num_bins=20, comm=comm,
        flip_g1=False, flip_g2=False, background_type= background_type,
        nside_coverage=32, min_distance_arcmin=1.0, max_distance_arcmin=60.0
    )

    if shear_flip_csv is not None:
        process_shear_sims(
            filament_h5, BG_data, output_shear_file=shear_flip_csv,
            k=1, num_bins=20, comm=comm,
            flip_g1=True, flip_g2=True, background_type=background_type,
            nside_coverage=32, min_distance_arcmin=1.0, max_distance_arcmin=60.0
        )










def sum_in_place(data, comm):
    """
    Use MPI to sum up the data from all the different processes in an array.

    Parameters
    ----------
    data : numpy.ndarray
        The data to sum up.

    comm : mpi4py.MPI.Comm
        The MPI communicator to use. If None, this function does nothing.
    """
    if comm is None:
        return

    import mpi4py.MPI

    if comm.Get_rank() == 0:
        comm.Reduce(mpi4py.MPI.IN_PLACE, data)
    else:
        comm.Reduce(data, None)