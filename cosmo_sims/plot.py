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

    input_dir = os.path.abspath(input_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[INFO] Output directory: {output_dir}\n")

    # GLOBAL containers across ALL categories
    ALL_GPLUS = []
    ALL_GX = []
    ALL_LABELS = []

    # Category-based containers:
    CATEGORY_DATA = {}

    # Loop through the top-level categories
    # e.g. Om_fixed, S8, S8_perp, sigma8_fixed
    for category in os.listdir(input_dir):
        cat_path = os.path.join(input_dir, category)
        if not os.path.isdir(cat_path):
            continue

        print(f"\n===== CATEGORY: {category} =====")
        CATEGORY_DATA[category] = {
            "labels": [],
            "gplus": [],
            "gx": []
        }

        # Walk entire tree under each category
        for root, dirs, files in os.walk(cat_path):

            # Only accept directories named "shear"
            if os.path.basename(root) != "shear":
                continue

            if "shear_p15.csv" not in files:
                continue

            csv_path = os.path.join(root, "shear_p15.csv")
            print(f"[INFO] Reading: {csv_path}")

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"  ERROR reading CSV: {e}")
                continue

            # Build a label: category_run_x_band_y
            rel_path = os.path.relpath(root, cat_path).split(os.sep)
            # rel_path = ["run_1", "band_0.1", "shear"]
            label = f"{category}_{'_'.join(rel_path[:-1])}"

            # Extract shear fields
            g_plus = df["g_plus"].values if "g_plus" in df.columns else None
            g_x    = df["g_x"].values    if "g_x" in df.columns else None

            # Store per-category
            CATEGORY_DATA[category]["labels"].append(label)
            CATEGORY_DATA[category]["gplus"].append(g_plus)
            CATEGORY_DATA[category]["gx"].append(g_x)

            # Store global
            ALL_LABELS.append(label)
            ALL_GPLUS.append(g_plus)
            ALL_GX.append(g_x)

            # ------------------------------
            # INDIVIDUAL PLOTS
            # ------------------------------
            if g_plus is not None:
                plt.figure(figsize=(7, 4.7))
                plt.plot(g_plus, linewidth=1.7)
                plt.xlabel("Index")
                plt.ylabel(r"$g_+$")
                plt.title(f"{label}: $g_+$")
                plt.tight_layout()
                out_file = os.path.join(output_dir, f"{label}_gplus.png")
                plt.savefig(out_file, dpi=250, bbox_inches="tight")
                plt.close()

            if g_x is not None:
                plt.figure(figsize=(7, 4.7))
                plt.plot(g_x, linewidth=1.7)
                plt.xlabel("Index")
                plt.ylabel(r"$g_{\times}$")
                plt.title(f"{label}: $g_{\times}$")
                plt.tight_layout()
                out_file = os.path.join(output_dir, f"{label}_gx.png")
                plt.savefig(out_file, dpi=250, bbox_inches="tight")
                plt.close()

        # ===============================
        # CATEGORY-LEVEL PLOTS
        # ===============================
        labels = CATEGORY_DATA[category]["labels"]
        gplus_list = CATEGORY_DATA[category]["gplus"]
        gx_list = CATEGORY_DATA[category]["gx"]

        # Category g+
        if any(g is not None for g in gplus_list):
            plt.figure(figsize=(8, 5.3))
            for g, lbl in zip(gplus_list, labels):
                if g is not None:
                    plt.plot(g, linewidth=1.5, label=lbl)
            plt.xlabel("Index")
            plt.ylabel(r"$g_+$")
            plt.title(f"{category}: All $g_+$")
            plt.legend(frameon=False)
            plt.tight_layout()
            out_file = os.path.join(output_dir, f"{category}_ALL_gplus.png")
            plt.savefig(out_file, dpi=300)
            plt.close()

        # Category g×
        if any(g is not None for g in gx_list):
            plt.figure(figsize=(8, 5.3))
            for g, lbl in zip(gx_list, labels):
                if g is not None:
                    plt.plot(g, linewidth=1.5, label=lbl)
            plt.xlabel("Index")
            plt.ylabel(r"$g_{\times}$")
            plt.title(f"{category}: All $g_{\times}$")
            plt.legend(frameon=False)
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
        plt.title("GLOBAL: All $g_+$ across all cosmologies")
        plt.legend(frameon=False, fontsize=7)
        plt.tight_layout()
        out_file = os.path.join(output_dir, f"GLOBAL_ALL_gplus.png")
        plt.savefig(out_file, dpi=300)
        plt.close()

    if any(g is not None for g in ALL_GX):
        plt.figure(figsize=(9, 6))
        for g, lbl in zip(ALL_GX, ALL_LABELS):
            if g is not None:
                plt.plot(g, linewidth=1.2, alpha=0.75, label=lbl)
        plt.xlabel("Index")
        plt.ylabel(r"$g_{\times}$")
        plt.title("GLOBAL: All $g_{\times}$ across all cosmologies")
        plt.legend(frameon=False, fontsize=7)
        plt.tight_layout()
        out_file = os.path.join(output_dir, f"GLOBAL_ALL_gx.png")
        plt.savefig(out_file, dpi=300)
        plt.close()

    print("\n[INFO] All plotting completed.\n")

# =====================================================================
if __name__ == "__main__":
    plot_all_shear("Cosmo_sim_ridges", "plots")
