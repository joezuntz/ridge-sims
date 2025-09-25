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

import h5py
import numpy as np


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None


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


def read_sim_background(bg_file, rows, comm=None):
    """Read background galaxies from simulated catalog (HDF5)."""
    with h5py.File(bg_file, "r") as f:
        if comm is None:
            s = slice(None)
        else:
            row_per_process = rows // comm.size
            s = slice(comm.rank * row_per_process, (comm.rank + 1) * row_per_process)

        bg_ra = f["RA"][s]
        bg_ra = (bg_ra + 180) % 360
        bg_dec = f["DEC"][s]
        g1 = f["G1"][s]
        g2 = f["G2"][s]
        z_true = f["Z_TRUE"][s]
        weights = f["weight"][s] if "weight" in f else np.ones_like(bg_ra)

    return bg_ra, bg_dec, g1, g2, z_true, weights


def read_DES_background(bg_file, comm=None):
    """Read background galaxies from DES-like catalog with 'background' group."""
    with h5py.File(bg_file, "r") as f:
        total_rows = f["background"]["ra"].shape[0]
        if comm is None:
            s = slice(None)
        else:
            row_per_process = total_rows // comm.size
            s = slice(comm.rank * row_per_process, (comm.rank + 1) * row_per_process)

        bg_ra = f["background"]["ra"][s]
        bg_ra = (bg_ra + 180) % 360
        bg_dec = f["background"]["dec"][s]
        g1 = f["background"]["g1"][s]
        g2 = f["background"]["g2"][s]
        weights = f["background"]["weight"][s]

    return bg_ra, bg_dec, g1, g2, None, weights  # DES has no z_true, return None


def load_background(bg_file, comm=None, rows=None, background_type=None):
    """Dispatch background reader based on catalog type."""
    if background_type == "sim":
        return read_sim_background(bg_file, rows, comm=comm)
    elif background_type == "DES":
        return read_DES_background(bg_file, comm=comm)
    else:
        raise ValueError(f"Unknown background_type: {background_type}")
		
		

def process_shear_sims(filament_file, bg_data, output_shear_file, k=1, num_bins=20, comm=comm,
                       flip_g1=False, flip_g2=False, background_type=None):  
    start_time = time.time()

    # Load filament data
    with h5py.File(filament_file, "r") as hdf:
        dataset = hdf["data"]
        ra_values = dataset["RA"][:]
        ra_values = ra_values
        dec_values = dataset["DEC"][:]
        labels = dataset["Filament_Label"][:]
        rows = dataset["RA"].size

    # === Only keep non-empty labels (and drop noise label -1 if present) ===
    unique_labels = [lab for lab in np.unique(labels) if lab != -1 and np.sum(labels == lab) > 0]
    print(f"Processing {len(unique_labels)} non-empty filaments (out of {len(np.unique(labels))} total labels)")


    bg_ra, bg_dec, g1_values, g2_values, z_true, weights = load_background(
    bg_data, comm=comm, rows=rows, background_type=background_type)
    
    
	
    # ========= SIGN-FLIP ==========
    if flip_g1:
        g1_values = -g1_values
    if flip_g2:
        g2_values = -g2_values
    # ==============================
        
#    Uncomment for non treated catalog
#    valid_mask = (
#        np.isfinite(bg_ra)
#        & np.isfinite(bg_dec)
#        & np.isfinite(g1_values)
#        & np.isfinite(g2_values)
#        & np.isfinite(weights)
#    )
#    bg_ra, bg_dec, g1_values, g2_values, weights = (
#        bg_ra[valid_mask],
#        bg_dec[valid_mask],
#        g1_values[valid_mask],
#        g2_values[valid_mask],
#        weights[valid_mask],
#    )

    bg_coords = np.radians(np.column_stack((bg_ra, bg_dec)))
    if bg_coords.shape[0] == 0:
        print(f"[rank {comm.rank if comm else 0}] WARNING: No background sources passed cuts! Skipping shear computation.")
        return
    max_distance = 0
    bin_sums_plus = np.zeros(num_bins)
    bin_sums_cross = np.zeros(num_bins)
    bin_weighted_distances = np.zeros(num_bins)
    bin_weights = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    for label in unique_labels:
        filament_mask = labels == label
        filament_coords = np.column_stack((ra_values[filament_mask], dec_values[filament_mask]))

        # Safety check
        if filament_coords.shape[0] == 0:
            print(f"Skipping empty filament label {label}")
            continue

        nbrs = NearestNeighbors(n_neighbors=1, metric="haversine").fit(filament_coords)
        distances, indices = nbrs.kneighbors(bg_coords)
        
#        # === TEMPORARY CODE TO CHECK DISTANCE DISTRIBUTION ===
#        print(f"Minimum distance found: {np.degrees(np.min(distances)) * 60:.2f} arcmin")
#        print(f"Maximum distance found: {np.degrees(np.max(distances)) * 60:.2f} arcmin")
    
#        # Find a reasonable percentile to set as your max bin
#        valid_distances = distances[np.where(distances > 0)]
#        max_bin_limit = np.percentile(valid_distances, 95)
#        print(f"95th percentile distance: {np.degrees(max_bin_limit) * 60:.2f} arcmin")
#        # === END TEMPORARY CODE ===
        
        matched_filament_points = filament_coords[indices[:, 0]]
        
        
        
        #########################################  Plots to check  ##############################################
#        # Plot the background coordinates in gray
#        plt.figure(figsize=(10, 8))
#        plt.scatter(np.radians(bg_ra), np.radians(bg_dec), s=1, c='gray', alpha=0.1)
#        # Plot the filaments color-coded by label
#        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
#        for i, label in enumerate(unique_labels):
#            filament_points_to_plot = np.column_stack((ra_values[labels == label], dec_values[labels == label]))
#            plt.scatter(filament_points_to_plot[:, 0], filament_points_to_plot[:, 1], s=5, color=colors[i], alpha=0.8)
        
#        plt.xlabel('RA [deg]')
#        plt.ylabel('Dec [deg]')
#        plt.title('Filaments and Background Galaxies')
        
#        # Ensure the output directory exists
#        plot_dir = 'filaments'
#        if not os.path.exists(plot_dir):
#            os.makedirs(plot_dir)
        
#        # Save the plot
#        plt.savefig(os.path.join(plot_dir, 'filaments_and_background.png'))
#        plt.close()
        
#        print(f"Filament plot saved to {os.path.join(plot_dir, 'filaments_and_background.png')}")
        
        #########################################################################################
        
        ###  This method is not wrong ####
        
        #delta_ra = matched_filament_points[:, 0] - bg_coords[:, 0]
        #delta_dec = matched_filament_points[:, 1] - bg_coords[:, 1]
        #phi = np.arctan2(delta_dec, delta_ra * np.cos(bg_coords[:, 1]))
        
        ### But this method is more precise ####
        
        # matched_filament_points and bg_coords are in radians right now.
        # So we need to convert them back to degrees for SkyCoord
        bg_sky = SkyCoord(ra=np.degrees(bg_coords[:, 0]) * u.deg,
                  dec=np.degrees(bg_coords[:, 1]) * u.deg)
        filament_sky = SkyCoord(ra=np.degrees(matched_filament_points[:, 0]) * u.deg,
                        dec=np.degrees(matched_filament_points[:, 1]) * u.deg)
        
        # Compute position angle of filament point relative to background galaxy
        phi = bg_sky.position_angle(filament_sky).rad  # radians, same as before
        
        

        g_plus = -g1_values * np.cos(2 * phi) + g2_values * np.sin(2 * phi)
        g_cross = g1_values * np.sin(2 * phi) - g2_values * np.cos(2 * phi)

        max_distance = max(max_distance, np.max(distances))

        min_ang_rad = np.radians(1 / 60)       # 1 arcmin
        max_ang_rad = np.radians(1.0)          # 1 degree
        bins = np.logspace(np.log10(min_ang_rad), np.log10(max_ang_rad), num_bins + 1)

        bin_indices = np.digitize(distances[:, 0], bins) - 1
        valid_bins = (bin_indices >= 0) & (bin_indices < num_bins)

        np.add.at(bin_sums_plus, bin_indices[valid_bins], weights[valid_bins] * g_plus[valid_bins])
        np.add.at(bin_sums_cross, bin_indices[valid_bins], weights[valid_bins] * g_cross[valid_bins])
        np.add.at(bin_weighted_distances, bin_indices[valid_bins], weights[valid_bins] * distances[valid_bins, 0])
        np.add.at(bin_weights, bin_indices[valid_bins], weights[valid_bins])
        np.add.at(bin_counts, bin_indices[valid_bins], 1)

    sum_in_place(bin_sums_plus, comm)
    sum_in_place(bin_sums_cross, comm)
    sum_in_place(bin_weighted_distances, comm)
    sum_in_place(bin_weights, comm)
    sum_in_place(bin_counts, comm)

    if comm is not None and comm.rank != 0:
        return

    weighted_g_plus = np.divide(bin_sums_plus, bin_weights, out=np.zeros_like(bin_sums_plus), where=bin_weights > 0)
    weighted_g_cross = np.divide(bin_sums_cross, bin_weights, out=np.zeros_like(bin_sums_cross), where=bin_weights > 0)
    weighted_real_distances = np.divide(bin_weighted_distances, bin_weights, out=np.zeros_like(bin_weighted_distances), where=bin_weights > 0)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    output_data = np.column_stack((bin_centers, weighted_real_distances, weighted_g_plus, weighted_g_cross, bin_counts, bin_weights))
    np.savetxt(output_shear_file, output_data, delimiter=",", header="Bin_Center,Weighted_Real_Distance,Weighted_g_plus,Weighted_g_cross,Counts,bin_weight", comments='')

    print(f"Shear processing completed in {time.time() - start_time:.2f} seconds.")

    # === PLOTTING IN LOG-LOG === # Uncomment for individual plots  
#    arcmin_centers = np.degrees(bin_centers) * 60
#    plt.figure()
#    plt.loglog(arcmin_centers, np.abs(weighted_g_plus), marker='o', label='|g_plus|')
#    plt.loglog(arcmin_centers, np.abs(weighted_g_cross), marker='x', label='|g_cross|')
#    plt.xlabel("Separation (arcmin)")
#    plt.ylabel("Shear amplitude")
#    plt.title("Tangential and Cross Shear")
#    plt.legend()
#    plt.grid(True, which='both', ls='--')
#    plot_file = output_shear_file.replace(".csv", "_shear_plot.png")
#    plt.savefig(plot_file, dpi=200)
#    plt.close()
#    print(f"Saved shear plot: {plot_file}")
    # ========================x


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