import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import h5py
import os
import matplotlib.image as mpimg
import matplotlib.patches as patches

############################################################
# === CONFIG===
############################################################
plt.rcParams.update({
    "figure.figsize": (6.8, 6.8),
    "figure.dpi": 100,
    "axes.linewidth": 1.6,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "font.family": "serif",
    "legend.frameon": False,
    "legend.fontsize": 12,
    "savefig.bbox": "tight",
})

############################################################
# === MST & DBSCAN functions ===
############################################################
def build_mst(points_rad, k=10):
    
    tree = KDTree(points_rad)
    distances, indices = tree.query(points_rad, k=k + 1)

    row, col, data = [], [], []
    for i in range(len(points_rad)):
        for j in range(1, k + 1):
            row.append(i)
            col.append(indices[i, j])
            data.append(distances[i, j])

    sparse_dist_matrix = coo_matrix((data, (row, col)),
                                    shape=(len(points_rad), len(points_rad)))
    mst_sparse = minimum_spanning_tree(sparse_dist_matrix).tocoo()

    G = nx.Graph()
    for i, j, weight in zip(mst_sparse.row, mst_sparse.col, mst_sparse.data):
        G.add_edge(int(i), int(j), weight=float(weight))
    return G


def detect_branch_points(mst):
    return [n for n, d in dict(mst.degree()).items() if d > 2]


def split_mst_at_branches(mst, branch_points):
    G = mst.copy()
    G.remove_nodes_from(branch_points)
    return list(nx.connected_components(G))


def segment_filaments_with_dbscan(points_rad, filament_segments,
                                  eps=0.02, min_samples=5):
    
    labels = np.full(len(points_rad), -1, dtype=int)
    cluster_id = 0

    for segment in filament_segments:
        segment = list(segment)
        segment_points = np.array([points_rad[idx] for idx in segment])

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        segment_labels = dbscan.fit_predict(segment_points)

        for i, idx in enumerate(segment):
            if segment_labels[i] != -1:
                labels[idx] = cluster_id + segment_labels[i]

        if len(segment_labels) > 0 and np.max(segment_labels) != -1:
            cluster_id += int(np.max(segment_labels)) + 1

    return labels

############################################################
# === Load ridge points ===
############################################################
def load_ridges_from_h5(path):
    """
    >>> UPDATED
    This now returns BOTH:
      - ridges_rad : analysis coordinates (radians)
      - ridges_deg : plotting coordinates (degrees)
    """
    with h5py.File(path, "r") as f:
        ridges = f["ridges"][:]

    dec_rad = ridges[:, 0].astype(float)
    ra_rad  = ridges[:, 1].astype(float)

    # >>> NEW:  check 
    max_abs = np.nanmax(np.abs(np.concatenate((dec_rad, ra_rad))))
    if max_abs > 2.0 * np.pi + 0.5:
        raise ValueError(
            "Input ridges appear inconsistent with radians."
        )

    # >>> NEW:  RA wrap in radians
    ra_rad = np.mod(ra_rad, 2.0 * np.pi)

    ridges_rad = np.column_stack((dec_rad, ra_rad))

    # >>> NEW: plotting-only copy in degrees
    ridges_deg = np.degrees(ridges_rad)
    ridges_deg[:, 1] = np.mod(ridges_deg[:, 1], 360.0)

    return ridges_rad, ridges_deg

############################################################
# === MAIN EXECUTION ===
############################################################
home_dir = "simulation_ridges_comparative_analysis_debug/normal_mesh_x2/band_0.1/Ridges_final_p15"
ridge_file = os.path.join(home_dir, "normal_run_1_ridges_p15.h5")

# >>> UPDATED: load both radian + degree representations
ridges_rad, ridges_deg = load_ridges_from_h5(ridge_file)

# ------------------------------------------------------------
# Region selection (RADIANS — unchanged numerically)
# ------------------------------------------------------------
ra_min, ra_max   = 3.35, 3.50
dec_min, dec_max = -1.0, -0.925

mask = (
    (ridges_rad[:, 1] >= ra_min) & (ridges_rad[:, 1] <= ra_max) &
    (ridges_rad[:, 0] >= dec_min) & (ridges_rad[:, 0] <= dec_max)
)

# >>> NEW: parallel rad/deg subsets with identical indexing
subset_rad = ridges_rad[mask]
subset_deg = ridges_deg[mask]

print(f"Selected {len(subset_rad)} points in region.")

# ------------------------------------------------------------
# MST + DBSCAN (RADIANS)
# ------------------------------------------------------------
mst = build_mst(subset_rad)
branches = detect_branch_points(mst)
segments = split_mst_at_branches(mst, branches)
labels = segment_filaments_with_dbscan(subset_rad, segments)

############################################################
# === OUTPUT DIRECTORY ===
############################################################
output_dir = "hyperparameter_test"
os.makedirs(output_dir, exist_ok=True)

############################################################
# === Helper for axis formatting ===
############################################################
def format_axes(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(top=True, right=True)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

############################################################
# (a) Ridge points — PLOTTED IN DEGREES
############################################################
fig, ax = plt.subplots()
ax.scatter(subset_deg[:, 1], subset_deg[:, 0], s=4, color="black")
ax.set_xlabel("RA [deg]")    
ax.set_ylabel("DEC [deg]")    
ax.invert_xaxis()
format_axes(ax)
fig.savefig(os.path.join(output_dir, "ridges-points.pdf"))
plt.close(fig)

############################################################
# (b) MST + branch points — edges mapped to DEGREES
############################################################
fig, ax = plt.subplots()

# >>> UPDATED: MST indices mapped to degree coordinates
for i, j in mst.edges():
    ax.plot([subset_deg[i, 1], subset_deg[j, 1]],
            [subset_deg[i, 0], subset_deg[j, 0]],
            color="gray", lw=0.7)

ax.scatter(subset_deg[:, 1], subset_deg[:, 0], s=6, color="black")

if len(branches) > 0:
    ax.scatter(subset_deg[branches, 1], subset_deg[branches, 0],
               s=14, color="red", zorder=5)

ax.set_xlabel("RA [deg]")   
ax.set_ylabel("DEC [deg]")   
ax.invert_xaxis() 
format_axes(ax)
fig.savefig(os.path.join(output_dir, "mst-branches.pdf"))
plt.close(fig)

############################################################
# (c) DBSCAN filaments — clustering in rad, plotting in deg
############################################################
fig, ax = plt.subplots()

for lab in np.unique(labels):
    m = labels == lab
    if lab == -1:
        ax.scatter(subset_deg[m, 1], subset_deg[m, 0],
                   s=3, color="lightgray")
    else:
        ax.scatter(subset_deg[m, 1], subset_deg[m, 0], s=4)

ax.set_xlabel("RA [deg]")    
ax.set_ylabel("DEC [deg]")  
ax.invert_xaxis()  
format_axes(ax)
fig.savefig(os.path.join(output_dir, "dbscan-filaments.pdf"))
plt.close(fig)

print("PDF plots saved.")

