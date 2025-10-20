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


#Une this code to run the temporary plot 


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
        return read_sim_background(bg_file, rows, comm=comm) #read_sim_background(bg_file, stride=100)
    elif background_type == "DES":
        return read_DES_background(bg_file, comm=comm)
    else:
        raise ValueError(f"Unknown background_type: {background_type}")
		

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
        return None, None, None, None, None

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
        bins = np.logspace(np.log10(min_ang_rad), np.log10(max_ang_rad), num_bins + 1)
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