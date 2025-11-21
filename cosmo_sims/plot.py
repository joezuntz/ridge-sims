import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
#   Publication-style params
# =====================================================================
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
    "xtick.top": True,
    "ytick.right": True,
})

# =====================================================================
def plot_all_shear(input_dir="Cosmo_sim_ridges", output_dir="plots"):

    # Absolute paths
    input_dir = os.path.abspath(input_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[INFO] Output directory: {output_dir}\n")

    # GLOBAL collectors
    ALL_GPLUS = []
    ALL_GX = []
    ALL_LABELS = []

    # CATEGORY collectors
    CATEGORY_DATA = {}

    # -----------------------------------------------------------------
    # Loop over cosmology categories
    # -----------------------------------------------------------------
    for category in os.listdir(input_dir):
        cat_path = os.path.join(input_dir, category)
        if not os.path.isdir(cat_path):
            continue

        print(f"\n===== CATEGORY: {category} =====")
        CATEGORY_DATA[category] = {"labels": [], "gplus": [], "gx": []}

        # -------------------------------------------------------------
        # Loop over runs inside each category
        # -------------------------------------------------------------
        for run in os.listdir(cat_path):
            run_path = os.path.join(cat_path, run)
            if not run.startswith("run_"):
                continue

            # ---------------------------------------------------------
            # Loop over bandwidths
            # ---------------------------------------------------------
            for band in os.listdir(run_path):
                band_path = os.path.join(run_path, band)
                if not band.startswith("band_"):
                    continue

                # Expected final directory
                ridge_path = os.path.join(band_path, "Ridges_final_p15", "shear")
                shear_file = os.path.join(ridge_path, "shear_p15.csv")

                # -----------------------------------------------------
                # Skip if CSV missing
                # -----------------------------------------------------
                if not os.path.exists(shear_file):
                    continue

                print(f"[INFO] Reading: {shear_file}")

                try:
                    df = pd.read_csv(shear_file)
                except Exception as e:
                    print(f"  ERROR reading CSV: {e}")
                    continue

                # Label format
                label = f"{category}_{run}_{band}"

                # Extract shear
                g_plus = df["g_plus"].values if "g_plus" in df.columns else None
                g_x    = df["g_x"].values    if "g_x" in df.columns else None

                # Store in category collectors
                CATEGORY_DATA[category]["labels"].append(label)
                CATEGORY_DATA[category]["gplus"].append(g_plus)
                CATEGORY_DATA[category]["gx"].append(g_x)

                # Store globally
                ALL_LABELS.append(label)
                ALL_GPLUS.append(g_plus)
                ALL_GX.append(g_x)

                # -----------------------------------------------------
                # INDIVIDUAL PLOTS
                # -----------------------------------------------------
                if g_plus is not None:
                    plt.figure(figsize=(7, 4.7))
                    plt.plot(g_plus, linewidth=1.7)
                    plt.xlabel("Index")
                    plt.ylabel(r"$g_+$")
                    plt.title(f"{label}  (g+)")
                    plt.tight_layout()
                    out_file = os.path.join(output_dir, f"{label}_gplus.png")
                    plt.savefig(out_file, dpi=250, bbox_inches="tight")
                    plt.close()

                if g_x is not None:
                    plt.figure(figsize=(7, 4.7))
                    plt.plot(g_x, linewidth=1.7)
                    plt.xlabel("Index")
                    plt.ylabel(r"$g_{\times}$")
                    plt.title(f"{label}  (g×)")
                    plt.tight_layout()
                    out_file = os.path.join(output_dir, f"{label}_gx.png")
                    plt.savefig(out_file, dpi=250, bbox_inches="tight")
                    plt.close()

        # =============================================================
        # PER-CATEGORY COMBINED PLOTS
        # =============================================================
        labels = CATEGORY_DATA[category]["labels"]
        gplus_list = CATEGORY_DATA[category]["gplus"]
        gx_list = CATEGORY_DATA[category]["gx"]

        # -- category g+ --
        if any(g is not None for g in gplus_list):
            plt.figure(figsize=(8, 5.3))
            for g, lbl in zip(gplus_list, labels):
                if g is not None:
                    plt.plot(g, linewidth=1.5, label=lbl)
            plt.xlabel("Index")
            plt.ylabel(r"$g_+$")
            plt.title(f"{category}: All g+ curves")
            plt.legend(frameon=False, fontsize=8)
            plt.tight_layout()
            out_file = os.path.join(output_dir, f"{category}_ALL_gplus.png")
            plt.savefig(out_file, dpi=300)
            plt.close()

        # -- category g× --
        if any(g is not None for g in gx_list):
            plt.figure(figsize=(8, 5.3))
            for g, lbl in zip(gx_list, labels):
                if g is not None:
                    plt.plot(g, linewidth=1.5, label=lbl)
            plt.xlabel("Index")
            plt.ylabel(r"$g_{\times}$")
            plt.title(f"{category}: All g× curves")
            plt.legend(frameon=False, fontsize=8)
            plt.tight_layout()
            out_file = os.path.join(output_dir, f"{category}_ALL_gx.png")
            plt.savefig(out_file, dpi=300)
            plt.close()

    # =================================================================
    # GLOBAL PLOTS ACROSS ALL CATEGORIES
    # =================================================================
    if any(g is not None for g in ALL_GPLUS):
        plt.figure(figsize=(9, 6))
        for g, lbl in zip(ALL_GPLUS, ALL_LABELS):
            if g is not None:
                plt.plot(g, linewidth=1.2, alpha=0.75, label=lbl)
        plt.xlabel("Index")
        plt.ylabel(r"$g_+$")
        plt.title("All cosmologies: g+")
        plt.legend(frameon=False, fontsize=6)
        plt.tight_layout()
        out_file = os.path.join(output_dir, "GLOBAL_ALL_gplus.png")
        plt.savefig(out_file, dpi=300)
        plt.close()

    if any(g is not None for g in ALL_GX):
        plt.figure(figsize=(9, 6))
        for g, lbl in zip(ALL_GX, ALL_LABELS):
            if g is not None:
                plt.plot(g, linewidth=1.2, alpha=0.75, label=lbl)
        plt.xlabel("Index")
        plt.ylabel(r"$g_{\times}$")
        plt.title("All cosmologies: g×")
        plt.legend(frameon=False, fontsize=6)
        plt.tight_layout()
        out_file = os.path.join(output_dir, "GLOBAL_ALL_gx.png")
        plt.savefig(out_file, dpi=300)
        plt.close()

    print("\n[INFO] All plotting completed.\n")


# =====================================================================
if __name__ == "__main__":
    plot_all_shear("Cosmo_sim_ridges", "plots")
