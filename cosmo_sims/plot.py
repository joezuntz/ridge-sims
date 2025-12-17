import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

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
    "xtick.major.size": 7,   # slightly longer ticks
    "ytick.major.size": 7,
})

# ------------------------------------------------------------
# Directories + config
# ------------------------------------------------------------
ROOT = "Cosmo_sim_ridges"
CATEGORIES = ["Om_fixed", "S8", "S8_perp", "sigma8_fixed"]
RUNS = [f"run_{i}" for i in range(1, 11)]
P = 15

OUTPUT_ROOT = os.path.abspath("plots_pdf")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"\nAll plots will be saved to: {OUTPUT_ROOT}\n")

# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def load_shear_file(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

# ------------------------------------------------------------
# Main plotting
# ------------------------------------------------------------
def plot_shear_all_categories():

    norm = Normalize(vmin=0, vmax=len(RUNS) - 1)
    cmap = cm.get_cmap("coolwarm")

    for cat in CATEGORIES:
        print(f"=== Category: {cat} ===")
        shear_list = []

        for run in RUNS:

            csv_path = os.path.join(
                ROOT, cat, run,
                "band_0.1", "Ridges_final_p15", "shear",
                f"shear_p{P:02d}.csv"
            )

            df = load_shear_file(csv_path)
            if df is None:
                print(f"  [missing] {csv_path}")
                continue

            shear_list.append((run, df))
            print(f"  [loaded] {csv_path}")

        if len(shear_list) == 0:
            print(f"  No usable shear files for {cat}.")
            continue

        fig_gplus, ax_gplus = plt.subplots(figsize=(8, 6))
        fig_gx, ax_gx = plt.subplots(figsize=(8, 6))

        for run_id, df in shear_list:

            # CSV columns (fixed)
            rad_col = "Weighted_Real_Distance"
            gplus_col = "Weighted_g_plus"
            gcross_col = "Weighted_g_cross"

            # Convert rad → arcmin
            arcmin_centers = np.degrees(df[rad_col].values) * 60.0

            # Color by run index (blue → red)
            run_index = int(run_id.split("_")[1]) - 1
            color = cmap(norm(run_index))

            # Plot
            ax_gplus.plot(arcmin_centers, df[gplus_col], alpha=0.6, color=color)
            ax_gx.plot(arcmin_centers, df[gcross_col], alpha=0.6, color=color)

        # ----------------------------------------------------
        # Colorbar 
        # ----------------------------------------------------
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        fig_gplus.colorbar(sm, ax=ax_gplus, pad=0.02)
        fig_gx.colorbar(sm, ax=ax_gx, pad=0.02)

        # ----------------------------------------------------
        # Format g+
        # ----------------------------------------------------
        ax_gplus.set_xscale("log")
        ax_gplus.set_title(r"{cat}: $\gamma_{+}$ ")
        ax_gplus.set_xlabel(r"$\theta [arcmin]$")
        ax_gplus.set_ylabel(r"$\gamma_{+}$")

        # ----------------------------------------------------
        # Format g×
        # ----------------------------------------------------
        ax_gx.set_xscale("log")
        ax_gx.set_title(r"{cat}:$\gamma_{times}$ ")
        ax_gx.set_xlabel(r"$\theta [arcmin]$")
        ax_gx.set_ylabel(r"$\gamma_{times}$")

        # ----------------------------------------------------
        # Save as PDF
        # ----------------------------------------------------
        fig_gplus.savefig(
            os.path.join(OUTPUT_ROOT, f"{cat}_gplus_all_runs.pdf"),
            bbox_inches="tight"
        )
        fig_gx.savefig(
            os.path.join(OUTPUT_ROOT, f"{cat}_gcross_all_runs.pdf"),
            bbox_inches="tight"
        )

        plt.close(fig_gplus)
        plt.close(fig_gx)

        print(f"  → saved {cat}_gplus_all_runs.pdf")
        print(f"  → saved {cat}_gcross_all_runs.pdf\n")

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    plot_shear_all_categories()
