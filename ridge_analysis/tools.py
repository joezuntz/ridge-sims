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
from sklearn.neighbors import BallTree
import dredge_mod 
from dredge_mod import filaments
import h5py
import numpy as np


################ Specific to Simulation Script 
##############################################
def transform_background(background_file, output_hdf5_file, i, chunk_size=100000):
    """Applies random rotation to shear values and saves the transformed background to an HDF5 file."""
    
    with h5py.File(background_file, "r") as file:
        background_group = file["background"]
        num_points = background_group["ra"].shape[0]

        with h5py.File(output_hdf5_file, "w") as out_file:
            # Create datasets in the output file with the same shape
            out_group = out_file.create_group("background")
            out_group.create_dataset("ra", shape=(num_points,), dtype="f8")
            out_group.create_dataset("dec", shape=(num_points,), dtype="f8")
            out_group.create_dataset("g1", shape=(num_points,), dtype="f8")
            out_group.create_dataset("g2", shape=(num_points,), dtype="f8")
            out_group.create_dataset("weight", shape=(num_points,), dtype="f8")

            # Process in chunks
            for start in range(0, num_points, chunk_size):
                end = min(start + chunk_size, num_points)

                # Read chunk
                bg_ra = background_group["ra"][start:end]
                bg_dec = background_group["dec"][start:end]
                g1_values = background_group["g1"][start:end]
                g2_values = background_group["g2"][start:end]
                weights = background_group["weight"][start:end]

                # Filter out NaN and Inf values
                valid_mask = np.isfinite(bg_ra) & np.isfinite(bg_dec) & np.isfinite(g1_values) & np.isfinite(g2_values) & np.isfinite(weights)

                # Apply filtering
                bg_ra, bg_dec = bg_ra[valid_mask], bg_dec[valid_mask]
                g1_values, g2_values = g1_values[valid_mask], g2_values[valid_mask]
                weights = weights[valid_mask]

                # Apply random shear rotation
                psi = np.random.uniform(0, 2 * np.pi, size=len(bg_ra))
                cos_2psi, sin_2psi = np.cos(2 * psi), np.sin(2 * psi)

                g1_transformed = cos_2psi * g1_values - sin_2psi * g2_values
                g2_transformed = sin_2psi * g1_values + cos_2psi * g2_values

                # Save transformed data
                out_group["ra"][start:start + len(bg_ra)] = bg_ra
                out_group["dec"][start:start + len(bg_dec)] = bg_dec
                out_group["g1"][start:start + len(g1_transformed)] = g1_transformed
                out_group["g2"][start:start + len(g2_transformed)] = g2_transformed
                out_group["weight"][start:start + len(weights)] = weights

    print(f"Transformed background saved to {output_hdf5_file}")


# just in case..
def transform_background_CSV(BG_data_clean, i):
    """Applies random rotation to shear values and saves the transformed background."""
    psi = np.random.uniform(0, 2 * np.pi, size=len(BG_data_clean))
    cos_2psi, sin_2psi = np.cos(2 * psi), np.sin(2 * psi)

    BG_data_clean["G1"], BG_data_clean["G2"] = (
        cos_2psi * BG_data_clean["G1"] - sin_2psi * BG_data_clean["G2"],
        sin_2psi * BG_data_clean["G1"] + cos_2psi * BG_data_clean["G2"]
    )

    filename = f"background_outputs/random_background_{i+1}.csv"
    BG_data_clean.to_csv(filename, index=False)
    #print(f"Saved: {filename}")



def generate_foreground(i, ra_min, ra_max, dec_min, dec_max, num_points, output_hdf5_file):
    """Generates random foreground ridges and saves them in HDF5 format."""
    
    # Calculate mean and standard deviation dynamically
    ra_mean, ra_std = (ra_max + ra_min) / 2, (ra_max - ra_min) / 4
    dec_mean, dec_std = (dec_max + dec_min) / 2, (dec_max - dec_min) / 4

    # Generate random RA/DEC values
    random_ra = np.random.normal(ra_mean, ra_std, num_points)
    random_dec = np.random.normal(dec_mean, dec_std, num_points)

    # Apply filtering to ensure values are within the defined boundaries
    random_ra = random_ra[(random_ra > ra_min) & (random_ra < ra_max)]
    random_dec = random_dec[(random_dec > dec_min) & (random_dec < dec_max)]

    # Ensure we get the required number of points
    while len(random_ra) < num_points or len(random_dec) < num_points:
        extra_ra = np.random.normal(ra_mean, ra_std, num_points - len(random_ra))
        extra_dec = np.random.normal(dec_mean, dec_std, num_points - len(random_dec))
        random_ra = np.concatenate((random_ra, extra_ra[(extra_ra > ra_min) & (extra_ra < ra_max)]))
        random_dec = np.concatenate((random_dec, extra_dec[(extra_dec > dec_min) & (extra_dec < dec_max)]))
        random_ra, random_dec = random_ra[:num_points], random_dec[:num_points]

    # Combine RA/DEC into coordinates
    random_coordinates = np.column_stack((random_ra, random_dec))

    # Save structured data to HDF5 file
    save_to_hdf5(random_coordinates, output_hdf5_file, dataset_name="foreground")
    return random_coordinates


###################### Common To All Scripts 
############################################
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

    
############## Saving and Loading Data #############
####################################################

# Just in case 
def save_filaments_to_csv(points, labels, filename):
    df = pd.DataFrame({"RA": points[:, 0], "DEC": points[:, 1], "Filament_Label": labels})
    df.to_csv(filename, index=False)
    #print(f"Filament data saved to {filename}")

def load_from_hdf5(filename, dataset_name="data"):
    """Load RA and DEC from an HDF5 file and return a NumPy array."""
    with h5py.File(filename, "r") as hdf:
        data = hdf[dataset_name][:]
    
    # Convert structured array to standard NumPy 2D array
    coordinates = np.column_stack((data["RA"], data["DEC"]))

    return coordinates


def save_to_hdf5(data, filename, dataset_name="data"):
    """Save RA and DEC as named columns"""
    dtype = [("RA", "f8"), ("DEC", "f8")]  # Define named columns
    structured_data = np.array([tuple(row) for row in data], dtype=dtype)  # Convert to structured array
    
    with h5py.File(filename, "w") as hdf:
        hdf.create_dataset(dataset_name, data=structured_data)

# There is an added label to each filament
def save_filaments_to_hdf5(ra_dec, labels, filename, dataset_name="data"):
    """Save RA, DEC, and Filament Labels"""
    dtype = [("RA", "f8"), ("DEC", "f8"), ("Filament_Label", "i8")]  
    structured_data = np.array(
        [(ra, dec, label) for (ra, dec), label in zip(ra_dec, labels)],
        dtype=dtype
    )

    with h5py.File(filename, "w") as hdf:
        hdf.create_dataset(dataset_name, data=structured_data)

        
# This is the structure of the DES data background 
def load_background_data(background_file):
    """Loads background data from an HDF5 file."""
    with h5py.File(background_file, "r") as file:
        bg_ra = file["background"]["ra"][:]
        bg_dec = file["background"]["dec"][:]
        g1_values = file["background"]["g1"][:]
        g2_values = file["background"]["g2"][:]
        weights = file["background"]["weight"][:]
    
    return bg_ra, bg_dec, g1_values, g2_values, weights


#this is only for an artificially created test set 
def load_filtered_background(background_file):
    """Loads filtered background data from an HDF5 file."""
    with h5py.File(background_file, "r") as file:
        background_group = file["background"]
        
        ra = background_group["ra"][:]
        dec = background_group["dec"][:]
        g1 = background_group["g1"][:]
        g2 = background_group["g2"][:]
        weight = background_group["weight"][:]
    
    return ra, dec, g1, g2, weight




def process_shear(filament_file, bg_data, output_shear_file, k=1, num_bins=20):
    """Optimized function to compute shear transformation and bin the results efficiently with minimal memory usage."""
    start_time = time.time()

    # Load filament data
    with h5py.File(filament_file, "r") as hdf:
        dataset = hdf["data"]
        ra_values = dataset["RA"][:]  # Right Ascension of filament points
        dec_values = dataset["DEC"][:]  # Declination of filament points
        labels = dataset["Filament_Label"][:]  # Labels identifying different filaments

    unique_labels = np.unique(labels)  # Get unique filament segment labels

    # Load background data
    with h5py.File(bg_data, "r") as file:
        background_group = file["background"]
        bg_ra = background_group["ra"][:]
        bg_dec = background_group["dec"][:]
        g1_values = background_group["g1"][:]
        g2_values = background_group["g2"][:]
        weights = background_group["weight"][:]

    # Filter valid background points to remove NaNs or infinite values
    valid_mask = np.isfinite(bg_ra) & np.isfinite(bg_dec) & np.isfinite(g1_values) & np.isfinite(g2_values) & np.isfinite(weights)
    bg_ra, bg_dec, g1_values, g2_values, weights = bg_ra[valid_mask], bg_dec[valid_mask], g1_values[valid_mask], g2_values[valid_mask], weights[valid_mask]
    
    bg_coords = np.radians(np.column_stack((bg_ra, bg_dec)))  # Convert background coordinates to radians

    # Define bin edges
    max_distance = 0
    bin_sums_plus = np.zeros(num_bins)
    bin_sums_cross = np.zeros(num_bins)
    bin_weighted_distances = np.zeros(num_bins)
    bin_weights = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    # Process each filament separately 
    for label in unique_labels:
        filament_mask = labels == label  # Select points belonging to the current filament
        filament_coords = np.radians(np.column_stack((ra_values[filament_mask], dec_values[filament_mask])))
        
        # Fit NearestNeighbors model for this filament
        nbrs = NearestNeighbors(n_neighbors=1, metric="haversine").fit(filament_coords)
        distances, indices = nbrs.kneighbors(bg_coords)  # Find closest filament point for each background point
        
        matched_filament_points = filament_coords[indices[:, 0]]  # Store matched filament points
        
        # Compute angular separation
        delta_ra = matched_filament_points[:, 0] - bg_coords[:, 0]
        delta_dec = matched_filament_points[:, 1] - bg_coords[:, 1]
        phi = np.arctan2(delta_dec, delta_ra * np.cos(bg_coords[:, 1]))
        
        # Compute shear components
        g_plus = -g1_values * np.cos(2 * phi) + g2_values * np.sin(2 * phi)
        g_cross = g1_values * np.sin(2 * phi) - g2_values * np.cos(2 * phi)
        
        max_distance = max(max_distance, np.max(distances))  # Update max distance for binning
        
        # Perform binning dynamically
        bins = np.linspace(0, max_distance * 1.05, num_bins + 1)  # Dynamic bin update
        bin_indices = np.digitize(distances[:, 0], bins) - 1  # Assign distances to bins
        valid_bins = (bin_indices >= 0) & (bin_indices < num_bins)

        np.add.at(bin_sums_plus, bin_indices[valid_bins], weights[valid_bins] * g_plus[valid_bins])
        np.add.at(bin_sums_cross, bin_indices[valid_bins], weights[valid_bins] * g_cross[valid_bins])
        np.add.at(bin_weighted_distances, bin_indices[valid_bins], weights[valid_bins] * distances[valid_bins, 0])
        np.add.at(bin_weights, bin_indices[valid_bins], weights[valid_bins])
        np.add.at(bin_counts, bin_indices[valid_bins], 1)

    # Compute final bin averages
    weighted_g_plus = np.divide(bin_sums_plus, bin_weights, out=np.zeros_like(bin_sums_plus), where=bin_weights > 0)
    weighted_g_cross = np.divide(bin_sums_cross, bin_weights, out=np.zeros_like(bin_sums_cross), where=bin_weights > 0)
    weighted_real_distances = np.divide(bin_weighted_distances, bin_weights, out=np.zeros_like(bin_weighted_distances), where=bin_weights > 0)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Compute bin centers for plotting

    output_data = np.column_stack((bin_centers, weighted_real_distances, weighted_g_plus, weighted_g_cross, bin_counts, bin_weights))
    
    # Save results to file
    np.savetxt(output_shear_file, output_data, delimiter=",", header="Bin_Center,Weighted_Real_Distance,Weighted_g_plus,Weighted_g_cross,Counts,bin_weight", comments='')
    
    print(f"Shear processing completed in {time.time() - start_time:.2f} seconds.")
