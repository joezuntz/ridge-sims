import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Publication-style parameters
# ------------------------------------------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})

# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------

SHEAR_ROOT = os.path.join("parameter_test", "shear_vs_meshsize")
OUTPUT_ROOT = os.path.abspath("hyperparam_plot")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"\nAll plots will be saved to: {OUTPUT_ROOT}\n")


# ------------------------------------------------------------
# Laod files
# ------------------------------------------------------------

def load_shear_files_from_meshsize():
    """
    Looks for:
        shear_vs_meshsize/shear_mesh_<mesh>.csv
    """
    out = []
    if not os.path.exists(SHEAR_ROOT):
        print(f"[FATAL] No such folder: {SHEAR_ROOT}")
        return out

    for f in sorted(os.listdir(SHEAR_ROOT)):
        if f.startswith("shear_mesh_") and f.endswith(".csv"):
            mesh = f.replace("shear_mesh_", "").replace(".csv", "")
            path = os.path.join(SHEAR_ROOT, f)

            try:
                df = pd.read_csv(path)
                out.append((mesh, df))
                print(f"Loaded: {path}")
            except Exception as e:
                print(f"[ERROR] Could not read: {path} → {e}")

    return out


# ------------------------------------------------------------
# Main plotting routine
# ------------------------------------------------------------

def plot_shear_meshsizes():

    shear_data = load_shear_files_from_meshsize()
    if len(shear_data) == 0:
        print("No shear files found.")
        return

    # Create figures
    fig_gplus, ax_gplus = plt.subplots(figsize=(8, 6))
    fig_gx, ax_gx = plt.subplots(figsize=(8, 6))

    # Fixed column names
    rad_col = "Weighted_Real_Distance"
    gplus_col = "Weighted_g_plus"
    gcross_col = "Weighted_g_cross"

    # Loop through meshes
    for mesh_value, df in shear_data:

        # Convert radian → arcmin
        theta_arcmin = np.degrees(df[rad_col].values) * 60.0

        # Plot profiles
        ax_gplus.plot(theta_arcmin, df[gplus_col],
                      alpha=0.7, label=f"mesh {mesh_value}")

        ax_gx.plot(theta_arcmin, df[gcross_col],
                   alpha=0.7, label=f"mesh {mesh_value}")

    # ----------------------------------------------------
    # Format g₊
    # ----------------------------------------------------
    ax_gplus.set_xscale("log")
    ax_gplus.set_title(r"$\gamma_{+}$ vs $\theta$ for different mesh sizes")
    ax_gplus.set_xlabel(r"$\theta$ [arcmin]")
    ax_gplus.set_ylabel(r"$\gamma_{+}$")
    ax_gplus.legend(frameon=False)

    # ----------------------------------------------------
    # Format g×
    # ----------------------------------------------------
    ax_gx.set_xscale("log")
    ax_gx.set_title(r"$\gamma_{times}$ vs $\theta$ for different mesh sizes")
    ax_gx.set_xlabel(r"$\theta$ [arcmin]")
    ax_gx.set_ylabel(r"$\gamma_{times}$")
    ax_gx.legend(frameon=False)

    # ----------------------------------------------------
    # Save plots
    # ----------------------------------------------------
    out_gplus = os.path.join(OUTPUT_ROOT, "all_mesh_sizes_gplus.png")
    out_gx = os.path.join(OUTPUT_ROOT, "all_mesh_sizes_gcross.png")

    fig_gplus.savefig(out_gplus, dpi=200, bbox_inches="tight")
    fig_gx.savefig(out_gx, dpi=200, bbox_inches="tight")

    plt.close(fig_gplus)
    plt.close(fig_gx)

    print(f"\nSaved:")
    print(" →", out_gplus)
    print(" →", out_gx, "\n")


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    plot_shear_meshsizes()
