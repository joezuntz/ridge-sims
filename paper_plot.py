import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import h5py
# === MST & DBSCAN functions ===
def build_mst(points, k=10):
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
    return [node for node, degree in dict(mst.degree()).items() if degree > 2]

def split_mst_at_branches(mst, branch_points):
    G = mst.copy()
    G.remove_nodes_from(branch_points)
    return list(nx.connected_components(G))

def segment_filaments_with_dbscan(points, filament_segments, eps=0.02, min_samples=5):
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


# === Load ridge points  ===
def load_ridges_from_h5(path):
    """This is a general function to load 'ridges' dataset from an h5 file and return array shape (N,2).
       Attempts to detect radians/degrees and applies canonicalization:
       returns array in degrees with columns [dec_deg, ra_deg_mod360].
    """
    with h5py.File(path, "r") as f:
        if "ridges" not in f:
            raise KeyError(f"'ridges' dataset not found in {path}")
        ridges = f["ridges"][:]
    ridges = np.asarray(ridges)
    if ridges.ndim != 2 or ridges.shape[1] < 2:
        raise ValueError("ridges array must be shape (N,2 or more) with dec,ra")

    dec = ridges[:, 0].astype(float)
    ra = ridges[:, 1].astype(float)

    # Detect radians vs degrees
    if np.nanmax(np.abs(np.concatenate((dec, ra)))) < 2.0:
        # radians -> convert to degrees
        dec = np.degrees(dec)
        ra = np.degrees(ra)

    # normalize RA into [0,360).
    ra = np.mod(ra, 360.0)

    return np.column_stack((dec, ra))




ridge_file = "WL_Mehraveh/ridge-sims/simulation_ridges_comparative_analysis/zero_err/band_0.1/Ridges_final_p15/zero_err_run_1_ridges_p15.h5"
ridges = load_ridges_from_h5(ridge_file)
# === 1. Select region ===
ra_min, ra_max = 3.2, 3.5   
dec_min, dec_max = -1.0, -0.8
mask = (ridges[:,1] >= ra_min) & (ridges[:,1] <= ra_max) & \
       (ridges[:,0] >= dec_min) & (ridges[:,0] <= dec_max)
subset = ridges[mask]
print(f"Selected {len(subset)} points in region.")

# === 2. Apply MST & DBSCAN ===
mst = build_mst(subset)
branches = detect_branch_points(mst)
segments = split_mst_at_branches(mst, branches)
labels = segment_filaments_with_dbscan(subset, segments, eps=0.02, min_samples=5)

# === 3. PLOTS ===



import os

# --- Define output folder ---
output_dir = os.path.join("simulation_ridges_comparative_analysis", "paper_plots")
os.makedirs(output_dir, exist_ok=True)


# (a) Ridge points
plt.figure(figsize=(6,6))
plt.scatter(subset[:,1], subset[:,0], s=4, color='black')
plt.xlabel("RA ")
plt.ylabel("DEC")
#plt.title("Ridge Points")
#plt.gca().invert_xaxis()  
plt.savefig(os.path.join(output_dir, "ridges_points.png"), dpi=300)

# (b) MST + branch points
plt.figure(figsize=(6,6))
for i, j in mst.edges():
    x = [subset[i,1], subset[j,1]]
    y = [subset[i,0], subset[j,0]]
    plt.plot(x, y, color='gray', lw=0.5, alpha=0.6)
plt.scatter(subset[:,1], subset[:,0], s=4, color='black', alpha=0.6)
plt.scatter(subset[branches,1], subset[branches,0], color='red', s=15, label='Branches')
plt.xlabel("RA")
plt.ylabel("DEC")
#plt.title("MST with Branch Points")
plt.legend()
#plt.gca().invert_xaxis()
plt.savefig(os.path.join(output_dir, "mst_branches.png"), dpi=300)

# (c) DBSCAN-labeled filaments
plt.figure(figsize=(6,6))
unique_labels = np.unique(labels)
for lab in unique_labels:
    mask = labels == lab
    if lab == -1:
        plt.scatter(subset[mask,1], subset[mask,0], color='lightgray', s=5, label='Noise')
    else:
        plt.scatter(subset[mask,1], subset[mask,0], s=5, label=f"Filament {lab}")
plt.xlabel("RA")
plt.ylabel("DEC")
#plt.title("Segmented filament")
plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.gca().invert_xaxis()
plt.savefig(os.path.join(output_dir, "dbscan_filaments.png"), dpi=300)
