import os
import numpy as np
import matplotlib.pyplot as plt


##############################################################

# -------------------NOISELESS-------------------------------





## --- Configuration ---
#base_sim_dir = "lhc_run_sims_zero_err_10"
#run_id = 1
#final_percentiles = [15]  # match generation script

## The directory structure mirrors the data generation code
#band_dir = "simulation_ridges_comparative_analysis_debug/zero_err_mesh_x2/band_0.1"
#shear_dir = os.path.join(band_dir, f"shear_test_data_{run_id}")
#plot_dir = os.path.join(shear_dir, "shear_plots")
#os.makedirs(plot_dir, exist_ok=True)


## ============================================================ #
## Helper function: plot shear
## ============================================================ #
#def plot_shear(case_label, shear_csv, plot_suffix):
#    print(f"\n=== Plotting shear for {case_label} ===")

#    if not os.path.exists(shear_csv):
#        print(f"[Warning] Missing shear file: {shear_csv}")
#        return

#    # --- Load shear data ---
#    data = np.loadtxt(shear_csv, delimiter=",", skiprows=1)
#    bin_center_rad = data[:, 0]
#    arcmin_centers = np.degrees(bin_center_rad) * 60.0
#    g_plus = data[:, 2]
#    g_cross = data[:, 3]

#    # --- Plot ---
#    plt.figure(figsize=(7, 5))
#    plt.plot(arcmin_centers, g_plus, 'o-', label=r"$g_+$")
#    plt.plot(arcmin_centers, g_cross, 'x-', label=r"$g_{\times}$")
#    plt.xscale("log")
#    plt.xlabel("Separation (arcmin)")
#    plt.ylabel("Shear")
#    plt.title(f"Shear profile ({case_label})")
#    plt.legend()
#    plt.grid(True, which="both", ls="--")
#    plt.tight_layout()

#    outpath = os.path.join(plot_dir, f"shear_only{plot_suffix}.png")
#    plt.savefig(outpath, dpi=200)
#    plt.close()
#    print(f"Saved: {outpath}")


## ============================================================ #
## Run for each case (normal + flipG1)
## ============================================================ #
#for fp in final_percentiles:
#    cases = {
#        "normal": ("", os.path.join(shear_dir, f"shear_p{fp:02d}.csv")),
#        "flipG1": ("_flipG1", os.path.join(shear_dir, f"shear_p{fp:02d}_flipG1.csv")),
#    }

#    for case, (suffix, shear_csv) in cases.items():
#        plot_shear(case, shear_csv, suffix)






###############################################################

# ------------------------ NOISY ------------------------------


# ==============================================================#
# Define output directory
# ==============================================================#
plot_dir = os.path.join(filament_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)


# ==============================================================#
# Full analysis for one case
# ==============================================================#
def run_analysis(case_label, shear_csv, noise_files, plot_suffix):
    print(f"\n=== Running analysis for {case_label} ===")

    # --- Load signal ---
    signal_data = np.loadtxt(shear_csv, delimiter=",", skiprows=1)
    bin_center_rad = signal_data[:, 0]
    arcmin_centers = np.degrees(bin_center_rad) * 60.0
    g_plus_signal = signal_data[:, 2]
    g_cross_signal = signal_data[:, 3]

    # --- Load noise realizations ---
    all_g_plus_noise = []
    all_g_cross_noise = []

    for nf in noise_files:
        if not os.path.exists(nf):
            print(f"[Warning] Missing noise file: {nf}")
            continue
        data = np.loadtxt(nf, delimiter=",", skiprows=1)
        all_g_plus_noise.append(data[:, 2])
        all_g_cross_noise.append(data[:, 3])

    if len(all_g_plus_noise) == 0 or len(all_g_cross_noise) == 0:
        print(f"[Error] No noise realizations found for {case_label}. Skipping.")
        return

    all_g_plus_noise = np.array(all_g_plus_noise)
    all_g_cross_noise = np.array(all_g_cross_noise)

    # --- Compute noise mean and std ---
    g_plus_noise_mean = np.mean(all_g_plus_noise, axis=0)
    g_cross_noise_mean = np.mean(all_g_cross_noise, axis=0)
    g_plus_noise_std = np.std(all_g_plus_noise, axis=0)
    g_cross_noise_std = np.std(all_g_cross_noise, axis=0)

    # --- Subtract noise ---
    g_plus_subtracted = g_plus_signal - g_plus_noise_mean
    g_cross_subtracted = g_cross_signal - g_cross_noise_mean

    # --- Covariance matrices from noise realizations ---
    cov_plus = np.cov(all_g_plus_noise, rowvar=False)
    cov_cross = np.cov(all_g_cross_noise, rowvar=False)

    # --- Regularize covariance if singular ---
    for name, cov in [('plus', cov_plus), ('cross', cov_cross)]:
        try:
            np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            print(f"[Warning] cov_{name} singular; regularizing.")
            cov += np.eye(cov.shape[0]) * 1e-12

    cov_plus_inv = np.linalg.inv(cov_plus)
    cov_cross_inv = np.linalg.inv(cov_cross)

    # --- Chi-square and detection significance ---
    d_plus = g_plus_subtracted
    d_cross = g_cross_subtracted
    dof = len(d_plus)
    chi2_plus = (d_plus @ cov_plus_inv @ d_plus)
    chi2_cross = (d_cross @ cov_cross_inv @ d_cross)
    pval_plus = 1 - stats.chi2.cdf(chi2_plus, dof)
    pval_cross = 1 - stats.chi2.cdf(chi2_cross, dof)
    sigma_plus = stats.norm.isf(pval_plus / 2.0)
    sigma_cross = stats.norm.isf(pval_cross / 2.0)

    print(f"[g_plus]  χ² = {chi2_plus:.2f}, dof = {dof}, p = {pval_plus:.2e}, ~ {sigma_plus:.2f}σ")
    print(f"[g_cross] χ² = {chi2_cross:.2f}, dof = {dof}, p = {pval_cross:.2e}, ~ {sigma_cross:.2f}σ")

    # ==============================================================#
    # Plots
    # ==============================================================#
    # 1. Noise only
    plt.figure(figsize=(7, 5))
    plt.errorbar(arcmin_centers, g_plus_noise_mean, yerr=g_plus_noise_std, fmt='o-', label=r"$g_+$ noise")
    plt.errorbar(arcmin_centers, g_cross_noise_mean, yerr=g_cross_noise_std, fmt='x-', label=r"$g_{\times}$ noise")
    plt.xscale("log")
    plt.xlabel("Separation (arcmin)")
    plt.ylabel("Shear (noise)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"noise_only{plot_suffix}.png"), dpi=200)
    plt.close()

    # 2. Signal vs noise-subtracted
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    ax[0].plot(arcmin_centers, g_plus_signal, 'k--', label="Raw $g_+$")
    ax[0].errorbar(arcmin_centers, g_plus_subtracted, yerr=g_plus_noise_std, fmt='o-', label="Signal - Noise")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Separation (arcmin)")
    ax[0].set_ylabel(r"$g_+$")
    ax[0].grid(True, which="both", ls="--")
    ax[0].legend()

    ax[1].plot(arcmin_centers, g_cross_signal, 'k--', label="Raw $g_{\\times}$")
    ax[1].errorbar(arcmin_centers, g_cross_subtracted, yerr=g_cross_noise_std, fmt='x-', label="Signal - Noise")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Separation (arcmin)")
    ax[1].set_ylabel(r"$g_{\times}$")
    ax[1].grid(True, which="both", ls="--")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"signal_vs_noise{plot_suffix}.png"), dpi=200)
    plt.close()

    # 3. Signal-minus-noise only
    plt.figure(figsize=(7, 5))
    plt.errorbar(arcmin_centers, g_plus_subtracted, yerr=g_plus_noise_std, fmt='o-', label=r"$g_+$ (signal - noise)")
    plt.errorbar(arcmin_centers, g_cross_subtracted, yerr=g_cross_noise_std, fmt='x-', label=r"$g_{\times}$ (signal - noise)")
    plt.xscale("log")
    plt.xlabel("Separation (arcmin)")
    plt.ylabel("Shear (signal - noise)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"signal_minus_noise{plot_suffix}.png"), dpi=200)
    plt.close()

    # 4. Noise vs signal-minus-noise
    plt.figure(figsize=(7, 5))
    plt.errorbar(arcmin_centers, g_plus_noise_mean, yerr=g_plus_noise_std, fmt='o--', label=r"$g_+$ noise")
    plt.errorbar(arcmin_centers, g_cross_noise_mean, yerr=g_cross_noise_std, fmt='x--', label=r"$g_{\times}$ noise")
    plt.errorbar(arcmin_centers, g_plus_subtracted, yerr=g_plus_noise_std, fmt='o-', label=r"$g_+$ (signal - noise)")
    plt.errorbar(arcmin_centers, g_cross_subtracted, yerr=g_cross_noise_std, fmt='x-', label=r"$g_{\times}$ (signal - noise)")
    plt.xscale("log")
    plt.xlabel("Separation (arcmin)")
    plt.ylabel("Shear")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"noise_vs_signal_minus_noise{plot_suffix}.png"), dpi=200)
    plt.close()

    # 5. Covariance heatmaps
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im0 = ax[0].imshow(cov_plus, origin="lower")
    ax[0].set_title(r"Covariance $g_+$")
    ax[0].set_xlabel("Bin")
    ax[0].set_ylabel("Bin")
    fig.colorbar(im0, ax=ax[0], shrink=0.8)

    im1 = ax[1].imshow(cov_cross, origin="lower")
    ax[1].set_title(r"Covariance $g_{\times}$")
    ax[1].set_xlabel("Bin")
    ax[1].set_ylabel("Bin")
    fig.colorbar(im1, ax=ax[1], shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"covariance_matrices{plot_suffix}.png"), dpi=200)
    plt.close()

    print(f"Finished analysis for {case_label}. Plots saved with suffix '{plot_suffix}'.")


# ==============================================================#
# Run for the test case
# ==============================================================#
fp = 15  # same percentile as above
shear_csv = os.path.join(filament_dir, f"shear_p{fp:02d}.csv")
shear_noise_dir = os.path.join(filament_dir, "shear")

noise_files = sorted([
    os.path.join(shear_noise_dir, f)
    for f in os.listdir(shear_noise_dir)
    if f.startswith(f"shear_noise_p{fp:02d}_") and f.endswith(".csv")
])

run_analysis("Run 1, percentile 15", shear_csv, noise_files, plot_suffix="_run1_p15")
