
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
############################################################

#os.chdir(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0, os.getcwd())


############################################################
############################################################




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


############################################################
# === Load ridge points  ===
############################################################
def load_ridges_from_h5(path):
    """Load 'ridges' dataset from an h5 file and return (N,2) array [dec, ra] in degrees."""
    with h5py.File(path, "r") as f:
        if "ridges" not in f:
            raise KeyError(f"'ridges' dataset not found in {path}")
        ridges = f["ridges"][:]
    ridges = np.asarray(ridges)
    if ridges.ndim != 2 or ridges.shape[1] < 2:
        raise ValueError("ridges array must be shape (N,2 or more) with dec,ra")

    dec = ridges[:, 0].astype(float)
    ra = ridges[:, 1].astype(float)

    if np.nanmax(np.abs(np.concatenate((dec, ra)))) < 2.0:
        dec = np.degrees(dec)
        ra = np.degrees(ra)

    ra = np.mod(ra, 360.0)
    return np.column_stack((dec, ra))


############################################################
# === MAIN EXECUTION ===
############################################################
home_dir = f"simulation_ridges_comparative_analysis_debug/normal/band_0.1/Ridges_final_p15"
ridge_file = os.path.join(home_dir,f"normal_run_1_ridges_p15.h5")
ridges = load_ridges_from_h5(ridge_file)

# === 1. Select region ===
ra_min, ra_max = 3.35, 3.50
dec_min, dec_max = -1.0, -0.925
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
output_dir = f"simulation_ridges_comparative_analysis_debug/normal/band_0.1/test_plots"
os.makedirs(output_dir, exist_ok=True)

# (a) Ridge points
plt.figure(figsize=(6,6))
plt.scatter(subset[:,1], subset[:,0], s=2, color='black')
plt.xlabel("RA")
plt.ylabel("DEC")
plt.savefig(os.path.join(output_dir, "ridges_points.png"), dpi=300)
plt.close()

# (b) MST + branch points
plt.figure(figsize=(6,6))
for i, j in mst.edges():
    x = [subset[i,1], subset[j,1]]
    y = [subset[i,0], subset[j,0]]
    plt.plot(x, y, color='gray', lw=0.5, alpha=1)
plt.scatter(subset[:,1], subset[:,0], s=4, color='black', alpha=1)
plt.scatter(subset[branches,1], subset[branches,0], color='red', s=2)
plt.xlabel("RA")
plt.ylabel("DEC")
plt.savefig(os.path.join(output_dir, "mst_branches.png"), dpi=300)
plt.close()

# (c) DBSCAN-labeled filaments
plt.figure(figsize=(6,6))
unique_labels = np.unique(labels)
for lab in unique_labels:
    mask = labels == lab
    if lab == -1:
        plt.scatter(subset[mask,1], subset[mask,0], color='lightgray', s=1)
    else:
        plt.scatter(subset[mask,1], subset[mask,0], s=1)
plt.xlabel("RA")
plt.ylabel("DEC")
plt.savefig(os.path.join(output_dir, "dbscan_filaments.png"), dpi=300)
plt.close()

print("Individual plots saved.")


############################################################
# === 4. FLOWCHART ===
############################################################
img_paths = [
    os.path.join(output_dir, "ridges_points.png"),
    os.path.join(output_dir, "mst_branches.png"),
    os.path.join(output_dir, "dbscan_filaments.png"),
]
titles = ["Ridge Points", "MST + Branches construction", "DBSCAN Filaments"]

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")

x_positions = [0, 4.2, 8.4]
y_pos = 0
box_w, box_h = 3.2, 3.2

for x, img_path, title in zip(x_positions, img_paths, titles):
    img = mpimg.imread(img_path)
    # Drop shadow
    shadow = patches.FancyBboxPatch(
        (x+0.1, y_pos-0.1), box_w, box_h,
        boxstyle="round,pad=0.02",
        linewidth=0, facecolor="gray", alpha=0.3, zorder=1
    )
    ax.add_patch(shadow)
    # Main box
    box = patches.FancyBboxPatch(
        (x, y_pos), box_w, box_h,
        boxstyle="round,pad=0.02",
        linewidth=1.2, edgecolor="black",
        facecolor="white", zorder=2
    )
    ax.add_patch(box)
    # Image inside
    ax.imshow(img, extent=(x, x+box_w, y_pos, y_pos+box_h), zorder=3)
    # Title
    ax.text(x + box_w/2, y_pos - 0.4, title, ha='center', va='top', fontsize=11, weight='semibold')

# Arrows
for i in range(2):
    x_start = x_positions[i] + box_w
    x_end = x_positions[i+1]
    y_mid = y_pos + box_h/2
    ax.annotate(
        "",
        xy=(x_end - 0.1, y_mid),
        xytext=(x_start + 0.1, y_mid),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="black", shrinkA=0, shrinkB=0),
    )

ax.set_xlim(-0.5, 12)
ax.set_ylim(-1, 4)
ax.set_aspect('equal')
plt.tight_layout()

flowchart_path = os.path.join(output_dir, "pipeline_overview.png")
plt.savefig(flowchart_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"flowchart saved to: {flowchart_path}")
