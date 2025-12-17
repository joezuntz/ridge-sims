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
    "xtick.major.size": 8,   # slightly longer ticks
    "ytick.major.size": 8,
})

# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------
ROOT = "Cosmo_sim_ridges"
CATEGORIES = ["Om_fixed", "S8", "S8_perp", "sigma8_fixed"]
RUNS = [f"run_{i}" for i in range(1, 11)]
P = 15

OUTPUT_ROOT = os.path.abspath("plots")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"\nAll plots will be saved to: {OUTPUT_ROOT}\n")

# ------------------------------------------------------------
# Load cosmology mapping
# ------------------------------------------------------------
mapping_path = "cosmo_run_mapping.csv"
mapping_df = pd.read_csv(mapping_path)

def get_param_label(category, run):
    row = mapping_df[(mapping_df.category == category) &
                     (mapping_df.run == run)]

    if len(row) == 0:
        return run

    row = row.iloc[0]

    if category == "Om_fixed":
        return rf"$\Omega_m = {row['Omega_m']:.3f}$"

    elif category == "S8":
        return rf"$S_8 = {row['S8']:.3f}$"

    elif category == "S8_perp":
        return rf"$S_8^\perp: \Omega_m = {row['Omega_m']:.3f}$"

    elif category == "sigma8_fixed":
        return rf"$\sigma_8 = {row['sigma8']:.3f}$"

    else:
        return ""

def get_param_value(category, run):
    """Numeric value used for colorbar."""
    row = mapping_df[(mapping_df.category == category) &
                     (mapping_df.run == run)]

    if len(row) == 0:
        return None

    row = row.iloc[0]

    if category == "Om_fixed":
        return row["Omega_m"]

    elif category == "S8":
        return row["S8"]

    elif category == "S8_perp":
        return row["Omega_m"]

    elif category == "sigma8_fixed":
        return row["sigma8"]

    else:
        return None

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

    for cat in CATEGORIES:
        print(f"=== Category: {cat} ===")
        shear_list = []
        param_values = []

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

            val = get_param_value(cat, run)
            if val is None:
                continue

            shear_list.append((df, val))
            param_values.append(val)

            print(f"  [loaded] {csv_path}")

        if len(shear_list) == 0:
            print(f"  No usable shear files for {cat}.")
            continue

        norm = Normalize(vmin=min(param_values), vmax=max(param_values))
        cmap = cm.get_cmap("coolwarm")

        fig_gplus, ax_gplus = plt.subplots(figsize=(8, 6))
        fig_gx, ax_gx = plt.subplots(figsize=(8, 6))

        for df, val in shear_list:

            rad_col = "Weighted_Real_Distance"
            gplus_col = "Weighted_g_plus"
            gcross_col = "Weighted_g_cross"

            arcmin_centers = np.degrees(df[rad_col].values) * 60.0
            color = cmap(norm(val))

            ax_gplus.plot(arcmin_centers, df[gplus_col], alpha=0.6, color=color)
            ax_gx.plot(arcmin_centers, df[gcross_col], alpha=0.6, color=color)

        # ----------------------------------------------------
        # Colorbar (instead of legend)
        # ----------------------------------------------------
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        fig_gplus.colorbar(sm, ax=ax_gplus, pad=0.02)
        fig_gx.colorbar(sm, ax=ax_gx, pad=0.02)

        # ----------------------------------------------------
        # Format g+
        # ----------------------------------------------------
        ax_gplus.set_xscale("log")
        ax_gplus.set_title(f"{cat}: " + r"$\gamma_{+}$")
        ax_gplus.set_xlabel(r"$\theta [arcmin]$")
        ax_gplus.set_ylabel(r" $\gamma_{+}$")

        # ----------------------------------------------------
        # Format g×
        # ----------------------------------------------------
        ax_gx.set_xscale("log")
        ax_gx.set_title(f"{cat}:"+ r"$\gamma_{times}$ ")
        ax_gx.set_xlabel(r"$\theta [arcmin]$")
        ax_gx.set_ylabel(r"$\gamma_{times}$")

        # ----------------------------------------------------
        # Save (PDF)
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
