import os
import pandas as pd
import matplotlib.pyplot as plt

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
#  publication style plotting param 
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

# ---------------------------------------------------------------------
def plot_all_shear(input_dir="Cosmo_sim_ridges", output_dir="plots"):

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Global containers for the “all-in-one” plots
    GLOBAL_GPLUS = []
    GLOBAL_GX = []
    GLOBAL_LABELS = []

    # -----------------------------------------------------------------
    # Scan shear directories
    # -----------------------------------------------------------------
    for root, dirs, files in os.walk(input_dir):

        if os.path.basename(root) != "shear":
            continue

        if "shear_p15.csv" not in files:
            continue

        csv_path = os.path.join(root, "shear_p15.csv")
        print(f"\nFound: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  FAILED to read {csv_path} → {e}")
            continue

        # label wrt directory structure
        rel_path = os.path.relpath(root, input_dir)
        label = rel_path.replace("/", "_")

        # store globally in plots
        if "g_plus" in df.columns:
            GLOBAL_GPLUS.append(df["g_plus"].values)
            GLOBAL_LABELS.append(label)

        if "g_x" in df.columns:
            GLOBAL_GX.append(df["g_x"].values)

        # -------------------------------------------------------------
        # Individual plots per directory (variant A)
        # -------------------------------------------------------------
        if "g_plus" in df.columns:
            plt.figure(figsize=(7,4.5))
            plt.plot(df["g_plus"].values, linewidth=1.4)
            plt.xlabel("Index")
            plt.ylabel(r"$g_{+}$")
            plt.title(f"g₊ — {label}")
            plt.tight_layout()
            out_file = os.path.join(output_dir, f"{label}_gplus.png")
            plt.savefig(out_file, dpi=200)
            plt.close()
            print(f"  Saved: {out_file}")

        if "g_x" in df.columns:
            plt.figure(figsize=(7,4.5))
            plt.plot(df["g_x"].values, linewidth=1.4)
            plt.xlabel("Index")
            plt.ylabel(r"$g_{\times}$")
            plt.title(f"g× — {label}")
            plt.tight_layout()
            out_file = os.path.join(output_dir, f"{label}_gx.png")
            plt.savefig(out_file, dpi=200)
            plt.close()
            print(f"  Saved: {out_file}")

    # -----------------------------------------------------------------
    # Variant B: All-directories combined plots
    # -----------------------------------------------------------------
    if GLOBAL_GPLUS:
        # ---- All g+ curves ----
        plt.figure(figsize=(8,5))
        for data, lbl in zip(GLOBAL_GPLUS, GLOBAL_LABELS):
            plt.plot(data, linewidth=1.4, label=lbl)
        plt.xlabel("Index")
        plt.ylabel(r"$g_{+}$")
        plt.title(r"All $g_{+}$ curves")
        plt.legend(frameon=False)
        plt.tight_layout()
        out_file = os.path.join(output_dir, "all_gplus_all_dirs.png")
        plt.savefig(out_file, dpi=220)
        plt.close()
        print(f"\nSaved global plot: {out_file}")

    if GLOBAL_GX:
        # ---- All g× curves ----
        plt.figure(figsize=(8,5))
        for data, lbl in zip(GLOBAL_GX, GLOBAL_LABELS):
            plt.plot(data, linewidth=1.4, label=lbl)
        plt.xlabel("Index")
        plt.ylabel(r"$g_{\times}$")
        plt.title(r"All $g_{\times}$ curves")
        plt.legend(frameon=False)
        plt.tight_layout()
        out_file = os.path.join(output_dir, "all_gx_all_dirs.png")
        plt.savefig(out_file, dpi=220)
        plt.close()
        print(f"Saved global plot: {out_file}")

    if GLOBAL_GPLUS and GLOBAL_GX:
        # ---- All g+ and g× overlapped ----
        plt.figure(figsize=(8,5))
        for data, lbl in zip(GLOBAL_GPLUS, GLOBAL_LABELS):
            plt.plot(data, linewidth=1.4, linestyle="-", label=lbl + "  (g+)")
        for data, lbl in zip(GLOBAL_GX, GLOBAL_LABELS):
            plt.plot(data, linewidth=1.4, linestyle="--", label=lbl + "  (g×)")
        plt.xlabel("Index")
        plt.ylabel(r"$g_{+},\, g_{\times}$")
        plt.title(r"All $g_{+}$ and $g_{\times}$ curves")
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        out_file = os.path.join(output_dir, "all_gplus_gx_all_dirs.png")
        plt.savefig(out_file, dpi=220)
        plt.close()
        print(f"Saved global plot: {out_file}")

    print("\nAll plotting completed.\n")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    plot_all_shear("Cosmo_sim_ridges", "plots")


