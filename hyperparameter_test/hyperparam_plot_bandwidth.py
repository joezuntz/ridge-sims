import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------
SHEAR_ROOT  = os.path.join("parameter_test", "shear_vs_bandwidth")
OUTPUT_ROOT = os.path.abspath("hyperparam_plot")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"\nAll plots will be saved to: {OUTPUT_ROOT}\n")

# ------------------------------------------------------------
# Load shear files
# ------------------------------------------------------------
def load_shear_files_from_bandwidth():
    """
    Looks for:
        shear_vs_bandwidth/shear_band_<bandwidth>.csv
    """
    out = []

    if not os.path.exists(SHEAR_ROOT):
        print(f"[FATAL] No such folder: {SHEAR_ROOT}")
        return out

    for f in sorted(os.listdir(SHEAR_ROOT)):
        if f.startswith("shear_band_") and f.endswith(".csv"):
            band = f.replace("shear_band_", "").replace(".csv", "")
            path = os.path.join(SHEAR_ROOT, f)

            try:
                df = pd.read_csv(path)
                out.append((float(band), df))
                print(f"Loaded: {path}")
            except Exception as e:
                print(f"[ERROR] Could not read: {path} → {e}")

    # Sort numerically by bandwidth
    out.sort(key=lambda x: x[0])
    return out

# ------------------------------------------------------------
# Main plotting routine
# ------------------------------------------------------------
def plot_shear_bandwidths():

    shear_data = load_shear_files_from_bandwidth()
    if len(shear_data) == 0:
        print("No shear files found.")
        return

    fig_gplus, ax_gplus = plt.subplots()
    fig_gx, ax_gx       = plt.subplots()

    # Fixed column names
    rad_col    = "Weighted_Real_Distance"
    gplus_col  = "Weighted_g_plus"
    gcross_col = "Weighted_g_cross"

    # ----------------------------------------------------
    # Colormap setup (bandwidth → color)
    # ----------------------------------------------------
    bands = [b for b, _ in shear_data]
    norm = colors.Normalize(vmin=min(bands), vmax=max(bands))
    cmap = cm.viridis
    sm   = cm.ScalarMappable(norm=norm, cmap=cmap)

    # ----------------------------------------------------
    # Plot
    # ----------------------------------------------------
    for band_value, df in shear_data:

        theta_arcmin = np.degrees(df[rad_col].values) * 60.0
        color = cmap(norm(band_value))

        ax_gplus.plot(theta_arcmin, df[gplus_col],
                      color=color, alpha=0.85)

        ax_gx.plot(theta_arcmin, df[gcross_col],
                   color=color, alpha=0.85)

    # ----------------------------------------------------
    # Format γ+
    # ----------------------------------------------------
    ax_gplus.set_xscale("log")
    ax_gplus.set_xlabel(r"$\theta$ [arcmin]")
    ax_gplus.set_ylabel(r"$\gamma_{+}$")
    ax_gplus.set_title(r"$\gamma_{+}$ vs $\theta$")

    cbar_gplus = fig_gplus.colorbar(sm, ax=ax_gplus, pad=0.02)
    cbar_gplus.set_label("Bandwidth")

    # ----------------------------------------------------
    # Format γ×
    # ----------------------------------------------------
    ax_gx.set_xscale("log")
    ax_gx.set_xlabel(r"$\theta$ [arcmin]")
    ax_gx.set_ylabel(r"$\gamma_{\times}$")
    ax_gx.set_title(r"$\gamma_{\times}$ vs $\theta$")

    cbar_gx = fig_gx.colorbar(sm, ax=ax_gx, pad=0.02)
    cbar_gx.set_label("Bandwidth")

    # ----------------------------------------------------
    # Save as PDF
    # ----------------------------------------------------
    out_gplus = os.path.join(OUTPUT_ROOT, "all_bandwidths_gplus.pdf")
    out_gx    = os.path.join(OUTPUT_ROOT, "all_bandwidths_gcross.pdf")

    fig_gplus.savefig(out_gplus)
    fig_gx.savefig(out_gx)

    plt.close(fig_gplus)
    plt.close(fig_gx)

    print("\nSaved:")
    print(" →", out_gplus)
    print(" →", out_gx, "\n")

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    plot_shear_bandwidths()
