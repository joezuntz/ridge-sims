import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors



# ------------------------------------------------------------
# Publication-style parameters
# ------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (8, 6.8),
    "figure.dpi": 100,

    "axes.linewidth": 1.6,
    "axes.labelsize": 15,
    "axes.titlesize": 15,

    # Major ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,

    # Minor ticks 
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

#================================================================
SHEAR_ROOT  = os.path.join("parameter_test", "shear_vs_bandwidth")
OUTPUT_ROOT = os.path.abspath("hyperparam_plot")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

RAD_COL   = "Weighted_Real_Distance"  
GPLUS_COL = "Weighted_g_plus"

ALL_CROSSINGS = True   # True: plot every zero crossing found; False: only the first one

def zero_crossings(theta, y):
    """Return all theta values where y crosses 0 by sign change, using linear interpolation in theta."""
    crossings = []
    for i in range(len(theta) - 1):
        y0, y1 = y[i], y[i+1]
        if y0 == 0.0:
            crossings.append(theta[i])
        elif y0 * y1 < 0.0:
            t = -y0 / (y1 - y0)  # fraction between points
            crossings.append(theta[i] + t * (theta[i+1] - theta[i]))
    return np.array(crossings, dtype=float)

def load_theta_y(path):
    df = pd.read_csv(path)
    theta = np.degrees(df[RAD_COL].to_numpy(float)) * 60.0  # arcmin
    y     = df[GPLUS_COL].to_numpy(float)
    m = np.isfinite(theta) & np.isfinite(y) & (theta > 0)
    theta, y = theta[m], y[m]
    s = np.argsort(theta)
    return theta[s], y[s]

# collect points
bw_vals = []
theta0_vals = []

bands = []
for f in sorted(os.listdir(SHEAR_ROOT)):
    if f.startswith("shear_band_") and f.endswith(".csv"):
        bands.append(float(f.replace("shear_band_", "").replace(".csv", "")))

norm = colors.Normalize(vmin=min(bands), vmax=max(bands))
cmap = cm.viridis

for f in sorted(os.listdir(SHEAR_ROOT)):
    if not (f.startswith("shear_band_") and f.endswith(".csv")):
        continue
    bw = float(f.replace("shear_band_", "").replace(".csv", ""))
    theta, y = load_theta_y(os.path.join(SHEAR_ROOT, f))

    z = zero_crossings(theta, y)
    if z.size == 0:
        continue

    if ALL_CROSSINGS:
        for th0 in z:
            bw_vals.append(bw)
            theta0_vals.append(th0)
    else:
        bw_vals.append(bw)
        theta0_vals.append(z.min())  # first crossing

bw_vals = np.array(bw_vals)
theta0_vals = np.array(theta0_vals)

# plot
fig, ax = plt.subplots()
sc = ax.scatter(
    bw_vals, theta0_vals,
    c=bw_vals, cmap=cmap, norm=norm,
    s=45, alpha=0.9, edgecolors="none"
)

ax.set_xlabel("Bandwidth")
ax.set_ylabel(r"separation $\theta_0$ [arcmin]")
ax.set_title(r"$\gamma_+(\theta)=0$")


ax.set_yscale("log")

cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("Bandwidth")

out = os.path.join(OUTPUT_ROOT, "zero_crossing_scatter_vs_bandwidth.pdf")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)

print("Saved:", out)
print("Plotted points:", len(theta0_vals))
