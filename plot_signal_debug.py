import os
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
base_sim_dir = "lhc_run_sims_zero_err_10"
run_id = 1
final_percentiles = [15]  # match generation script

# The directory structure mirrors the data generation code
band_dir = "simulation_ridges_comparative_analysis_debug/zero_err_mesh_x2/band_0.1"
shear_dir = os.path.join(band_dir, f"shear_test_data_{run_id}")
plot_dir = os.path.join(shear_dir, "shear_plots")
os.makedirs(plot_dir, exist_ok=True)


# ============================================================ #
# Helper function: plot shear
# ============================================================ #
def plot_shear(case_label, shear_csv, plot_suffix):
    print(f"\n=== Plotting shear for {case_label} ===")

    if not os.path.exists(shear_csv):
        print(f"[Warning] Missing shear file: {shear_csv}")
        return

    # --- Load shear data ---
    data = np.loadtxt(shear_csv, delimiter=",", skiprows=1)
    bin_center_rad = data[:, 0]
    arcmin_centers = np.degrees(bin_center_rad) * 60.0
    g_plus = data[:, 2]
    g_cross = data[:, 3]

    # --- Plot ---
    plt.figure(figsize=(7, 5))
    plt.plot(arcmin_centers, g_plus, 'o-', label=r"$g_+$")
    plt.plot(arcmin_centers, g_cross, 'x-', label=r"$g_{\times}$")
    plt.xscale("log")
    plt.xlabel("Separation (arcmin)")
    plt.ylabel("Shear")
    plt.title(f"Shear profile ({case_label})")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()

    outpath = os.path.join(plot_dir, f"shear_only{plot_suffix}.png")
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved: {outpath}")


# ============================================================ #
# Run for each case (normal + flipG1)
# ============================================================ #
for fp in final_percentiles:
    cases = {
        "normal": ("", os.path.join(shear_dir, f"shear_p{fp:02d}.csv")),
        "flipG1": ("_flipG1", os.path.join(shear_dir, f"shear_p{fp:02d}_flipG1.csv")),
    }

    for case, (suffix, shear_csv) in cases.items():
        plot_shear(case, shear_csv, suffix)
