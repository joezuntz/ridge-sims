import os
import pandas as pd
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
# Directories + config
# ------------------------------------------------------------

# Data root
ROOT = "Cosmo_sim_ridges"

# Categories (top-level dirs under ROOT)
CATEGORIES = ["Om_fixed", "S8", "S8_perp", "sigma8_fixed"]

# Runs
RUNS = [f"run_{i}" for i in range(1, 11)]
P = 15      # percentile → shear_p15.csv


# ------------------------------------------------------------
# Output directory (relative to where you run the script)
# ------------------------------------------------------------
OUTPUT_ROOT = os.path.abspath("plots")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"\nAll plots will be saved to: {OUTPUT_ROOT}\n")


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

def load_shear_file(path):
    """Load shear CSV; returns None if missing or unreadable."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ------------------------------------------------------------
# Main plotting routine
# ------------------------------------------------------------

def plot_shear_all_categories():

    for cat in CATEGORIES:
        print(f"=== Category: {cat} ===")
        shear_list = []   # list of (run_id, df)

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

        # ----------------------------------------------------
        # If nothing loaded for this category, skip
        # ----------------------------------------------------
        if len(shear_list) == 0:
            print(f"  No usable shear files for {cat}.")
            continue

        # ----------------------------------------------------
        # Prepare two separate figures: g+ and g×
        # ----------------------------------------------------
        fig_gplus, ax_gplus = plt.subplots(figsize=(8, 6))
        fig_gx, ax_gx = plt.subplots(figsize=(8, 6))

        for run_id, df in shear_list:

            # Identify columns automatically
            # Radius
            if "R" in df.columns:
                rad_col = "R"
            else:
                rad_col = [c for c in df.columns if "R" in c][0]

            # g+
            gplus_candidates = [c for c in df.columns if "gplus" in c.lower() or "g_plus" in c.lower()]
            if len(gplus_candidates) == 0:
                raise ValueError(f"Could not find g+ column in {csv_path}")
            gplus_col = gplus_candidates[0]

            # g×
            gcross_candidates = [c for c in df.columns if "x" in c.lower()]
            if len(gcross_candidates) == 0:
                raise ValueError(f"Could not find gx column in {csv_path}")
            gcross_col = gcross_candidates[0]

            # Plot
            ax_gplus.plot(df[rad_col], df[gplus_col], alpha=0.6, label=run_id)
            ax_gx.plot(df[rad_col], df[gcross_col], alpha=0.6, label=run_id)

        # ----------------------------------------------------
        # Format g+
        # ----------------------------------------------------
        ax_gplus.set_title(f"{cat}: g₊ (all runs)")
        ax_gplus.set_xlabel("R")
        ax_gplus.set_ylabel("g₊")
        ax_gplus.legend(frameon=False)

        # ----------------------------------------------------
        # Format g×
        # ----------------------------------------------------
        ax_gx.set_title(f"{cat}: g× (all runs)")
        ax_gx.set_xlabel("R")
        ax_gx.set_ylabel("g×")
        ax_gx.legend(frameon=False)

        # ----------------------------------------------------
        # Save both figures into ./plots/
        # ----------------------------------------------------
        fig_gplus.savefig(
            os.path.join(OUTPUT_ROOT, f"{cat}_gplus_all_runs.png"),
            dpi=200, bbox_inches="tight"
        )
        fig_gx.savefig(
            os.path.join(OUTPUT_ROOT, f"{cat}_gcross_all_runs.png"),
            dpi=200, bbox_inches="tight"
        )

        plt.close(fig_gplus)
        plt.close(fig_gx)

        print(f"  → saved {cat}_gplus_all_runs.png")
        print(f"  → saved {cat}_gcross_all_runs.png\n")


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    plot_shear_all_categories()
