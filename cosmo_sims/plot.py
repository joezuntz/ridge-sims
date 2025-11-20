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

    # Output directory relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[INFO] Output directory resolved to: {output_dir}\n")

    # Global containers for all-directory plots
    GLOBAL_GPLUS = []
    GLOBAL_GX = []
    GLOBAL_LABELS = []

    # =================================================================
    # Walk over all directories
    # =================================================================
    for root, dirs, files in os.walk(input_dir):

        if os.path.basename(root) != "shear":
            continue
        if "shear_p15.csv" not in files:
            continue

        csv_path = os.path.join(root, "shear_p15.csv")
        print(f"Reading: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  ERROR reading file: {e}")
            continue

        rel_path = os.path.relpath(root, input_dir)
        label = rel_path.replace("/", "_")

        # Store globally
        g_plus = df["g_plus"].values if "g_plus" in df.columns else None
        g_x    = df["g_x"].values    if "g_x" in df.columns else None

        GLOBAL_LABELS.append(label)
        GLOBAL_GPLUS.append(g_plus)
        GLOBAL_GX.append(g_x)

        # =============================================================
        # Individual g+ plot
        # =============================================================
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
            print(f"  Saved {out_file}")

        # =============================================================
        # Individual g× plot
        # =============================================================
        if g_x is not None:
            plt.figure(figsize=(7, 4.7))
            plt.plot(g_x, linewidth=1.7)
            plt.xlabel("Index")
            plt.ylabel(r"$g_{\times}$")
            plt.title(label + r": $g_{\times}$")
            plt.tight_layout()
            out_file = os.path.join(output_dir, f"{label}_gx.png")
            plt.savefig(out_file, dpi=250, bbox_inches="tight")
            plt.close()
            print(f"  Saved {out_file}")


    # =================================================================
    # GLOBAL PLOTS
    # =================================================================
    # -------------------------
    # All g+
    # -------------------------
    if any(x is not None for x in GLOBAL_GPLUS):
        plt.figure(figsize=(8, 5.3))
        for g, lbl in zip(GLOBAL_GPLUS, GLOBAL_LABELS):
            if g is not None:
                plt.plot(g, linewidth=1.5, label=lbl)
        plt.xlabel("Index")
        plt.ylabel(r"$g_+$")
        plt.title(r"All $g_+$ curves")
        plt.legend(frameon=False)
        plt.tight_layout()
        out_file = os.path.join(output_dir, "ALL_gplus.png")
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nSaved global: {out_file}")

    # -------------------------
    # All g×
    # -------------------------
    if any(x is not None for x in GLOBAL_GX):
        plt.figure(figsize=(8, 5.3))
        for g, lbl in zip(GLOBAL_GX, GLOBAL_LABELS):
            if g is not None:
                plt.plot(g, linewidth=1.5, label=lbl)
        plt.xlabel("Index")
        plt.ylabel(r"$g_{\times}$")
        plt.title(r"All $g_{\times}$ curves")
        plt.legend(frameon=False)
        plt.tight_layout()
        out_file = os.path.join(output_dir, "ALL_gx.png")
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved global: {out_file}")

    # -------------------------
    # Overlapped g+ and g×
    # -------------------------
    if any(x is not None for x in GLOBAL_GPLUS) and any(x is not None for x in GLOBAL_GX):
        plt.figure(figsize=(8, 5.3))

        for g, lbl in zip(GLOBAL_GPLUS, GLOBAL_LABELS):
            if g is not None:
                plt.plot(g, linewidth=1.5, linestyle="-", label=lbl + " (g+)")

        for g, lbl in zip(GLOBAL_GX, GLOBAL_LABELS):
            if g is not None:
                plt.plot(g, linewidth=1.5, linestyle="--", label = lbl + r"  $(g_{\times})$")

        plt.xlabel("Index")
        plt.ylabel(r"$g_+,\, g_{\times}$")
        plt.title(r"All $g_+$ and $g_{\times}$ curves")
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        out_file = os.path.join(output_dir, "ALL_gplus_gx_overlapped.png")
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved global: {out_file}")

    print("\nAll plotting completed.\n")


# =====================================================================
if __name__ == "__main__":
    plot_all_shear("Cosmo_sim_ridges", "plots")
