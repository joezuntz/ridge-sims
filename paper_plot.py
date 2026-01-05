import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import h5py
import os, sys
import matplotlib.image as mpimg
import matplotlib.patches as patches

############################################################
# === PUBLICATION STYLE CONFIG===
############################################################
plt.rcParams.update({
    "figure.figsize": (6.8, 6.8),   
    "figure.dpi": 100,

    "axes.linewidth": 1.6,
    "axes.labelsize": 15,
    "axes.titlesize": 15,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,

    "font.family": "serif",

    "legend.frameon": False,
    "legend.fontsize": 12,

    "savefig.bbox": "tight",
})

############################################################
# === MST & DBSCAN functions ===
############################################################
def build_mst(points, k=10):
    tree = KDTree(points)
    distances, indices = tree.query(points, k=k+1)
    row, col, data = [], [], []
    for i in range(len(points)):
        for j in range(1, k+1):
            row.append(i)
            col.append(indices[i, j])
            data.append(distances[i, j])
    sparse_dist_matrix = coo_matrix((data, (row, col)),
                                    shape=(len(points), len(points)))
    mst_sparse = minimum_spanning_tree(sparse_dist_matrix).tocoo()
    G = nx.Graph()
    for i, j, weight in zip(mst_sparse.row, mst_sparse.col, mst_sparse.data):
        G.add_edge(int(i), int(j), weight=weight)
    return G

def detect_branch_points(mst):
    return [n for n, d in dict(mst.degree()).items() if d > 2]

def split_mst_at_branches(mst, branch_points):
    G = mst.copy()
    G.remove_nodes_from(branch_points)
    return list(nx.connected_components(G))

def segment_filaments_with_dbscan(points, filament_segments,
                                  eps=0.02, min_samples=5):
    labels = np.full(len(points), -1)
    cluster_id = 0
    for segment in filament_segments:
        segment_points = np.array([points[idx] for idx in segment])
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        segment_labels = dbscan.fit_predict(segment_points)
        for i, idx in enumerate(segment):
            if segment_labels[i] != -1:
                labels[idx] = cluster_id + segment_labels[i]
        if len(segment_labels) > 0:
            cluster_id += max(segment_labels) + 1
    return labels

############################################################
# === Load ridge points ===
############################################################
def load_ridges_from_h5(path):
    with h5py.File(path, "r") as f:
        ridges = f["ridges"][:]

    dec = ridges[:, 0].astype(float)
    ra  = ridges[:, 1].astype(float)

    if np.nanmax(np.abs(np.concatenate((dec, ra)))) < 2.0:
        dec = np.degrees(dec)
        ra  = np.degrees(ra)

    ra = np.mod(ra, 360.0)
    return np.column_stack((dec, ra))

############################################################
# === MAIN EXECUTION ===
############################################################
home_dir = "simulation_ridges_comparative_analysis_debug/normal_mesh_x2/band_0.1/Ridges_final_p15"
ridge_file = os.path.join(home_dir, "normal_run_1_ridges_p15.h5")
ridges = load_ridges_from_h5(ridge_file)

# Region selection
ra_min, ra_max   = 3.35, 3.50
dec_min, dec_max = -1.0, -0.925

mask = (
    (ridges[:,1] >= ra_min) & (ridges[:,1] <= ra_max) &
    (ridges[:,0] >= dec_min) & (ridges[:,0] <= dec_max)
)
subset = ridges[mask]
print(f"Selected {len(subset)} points in region.")

# MST + DBSCAN
mst = build_mst(subset)
branches = detect_branch_points(mst)
segments = split_mst_at_branches(mst, branches)
labels = segment_filaments_with_dbscan(subset, segments)

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
# (a) Ridge points
############################################################
fig, ax = plt.subplots()
ax.scatter(subset[:,1], subset[:,0], s=4, color="black")
ax.set_xlabel("RA")
ax.set_ylabel("DEC")
format_axes(ax)
fig.savefig(os.path.join(output_dir, "ridges_points.pdf"))
plt.close(fig)

############################################################
# (b) MST + branch points
############################################################
fig, ax = plt.subplots()
for i, j in mst.edges():
    ax.plot([subset[i,1], subset[j,1]],
            [subset[i,0], subset[j,0]],
            color="gray", lw=0.7)

ax.scatter(subset[:,1], subset[:,0], s=6, color="black")
ax.scatter(subset[branches,1], subset[branches,0],
           s=14, color="red", zorder=5)

ax.set_xlabel("RA")
ax.set_ylabel("DEC")
format_axes(ax)
fig.savefig(os.path.join(output_dir, "mst_branches.pdf"))
plt.close(fig)

############################################################
# (c) DBSCAN filaments
############################################################
fig, ax = plt.subplots()
for lab in np.unique(labels):
    m = labels == lab
    if lab == -1:
        ax.scatter(subset[m,1], subset[m,0],
                   s=3, color="lightgray")
    else:
        ax.scatter(subset[m,1], subset[m,0], s=4)

ax.set_xlabel("RA")
ax.set_ylabel("DEC")
format_axes(ax)
fig.savefig(os.path.join(output_dir, "dbscan_filaments.pdf"))
plt.close(fig)

print("PDF plots saved.")

############################################################
# === FLOWCHART (PDF) ===
############################################################
img_paths = [
    os.path.join(output_dir, "ridges_points.pdf"),
    os.path.join(output_dir, "mst_branches.pdf"),
    os.path.join(output_dir, "dbscan_filaments.pdf"),
]

titles = [
    "Ridge Points",
    "MST + Branch Construction",
    "DBSCAN Filaments",
]

fig, ax = plt.subplots(figsize=(4.2, 10))
ax.axis("off")

x0 = 0
y_positions = [8.4, 4.2, 0.0]
box_w, box_h = 3.4, 3.4

for y, path, title in zip(y_positions, img_paths, titles):
    img = mpimg.imread(path)

    box = patches.FancyBboxPatch(
        (x0, y), box_w, box_h,
        boxstyle="round,pad=0.02",
        linewidth=1.4,
        edgecolor="black",
        facecolor="white"
    )
    ax.add_patch(box)
    ax.imshow(img, extent=(x0, x0+box_w, y, y+box_h))
    ax.text(x0 + box_w/2, y - 0.45, title,
            ha="center", va="top",
            fontsize=12, weight="semibold")

for i in range(2):
    ax.annotate("",
        xy=(x0+box_w/2, y_positions[i+1]+box_h+0.2),
        xytext=(x0+box_w/2, y_positions[i]-0.6),
        arrowprops=dict(arrowstyle="->", lw=2)
    )

ax.set_xlim(-0.5, box_w + 0.5)
ax.set_ylim(-1, 12)
ax.set_aspect("equal")

fig.savefig(os.path.join(output_dir, "pipeline_overview.pdf"))
plt.close(fig)

print("Pipeline overview PDF saved.")
