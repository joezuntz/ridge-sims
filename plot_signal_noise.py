import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- Configuration ---
filament_dir = "example_zl04_mesh5e5/filaments"
noise_shear_dir = "example_zl04_mesh5e5/noise/shear"
plot_dir = "example_zl04_mesh5e5/shear_plots"
os.makedirs(plot_dir, exist_ok=True)

final_percentile = 15
num_realizations = 300

# ==============================================================#
# Run full analysis for one case
# ==============================================================#
def run_analysis(case_label, shear_csv, noise_files, plot_suffix):
    print(f"\n=== Running analysis for {case_label} ===")

    # --- Load real signal ---
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

    # guard against singular covariance
    try:
        cov_plus_inv = np.linalg.inv(cov_plus)
    except np.linalg.LinAlgError:
        print("[Error] cov_plus is singular; adding small regularization.")
        cov_plus += np.eye(cov_plus.shape[0]) * 1e-12
        cov_plus_inv = np.linalg.inv(cov_plus)

    try:
        cov_cross_inv = np.linalg.inv(cov_cross)
    except np.linalg.LinAlgError:
        print("[Error] cov_cross is singular; adding small regularization.")
        cov_cross += np.eye(cov_cross.shape[0]) * 1e-12
        cov_cross_inv = np.linalg.inv(cov_cross)

    # --- Data vectors ---
    d_plus = g_plus_subtracted
    d_cross = g_cross_subtracted

    # --- Model (zero signal) ---
    m_plus = np.zeros_like(d_plus)
    m_cross = np.zeros_like(d_cross)

    # --- Chi-square ---
    chi2_plus = (d_plus - m_plus) @ cov_plus_inv @ (d_plus - m_plus)
    chi2_cross = (d_cross - m_cross) @ cov_cross_inv @ (d_cross - m_cross)
    dof = len(d_plus)

    # --- p-values & sigma ---
    pval_plus = 1 - stats.chi2.cdf(chi2_plus, dof)
    pval_cross = 1 - stats.chi2.cdf(chi2_cross, dof)
    # two-tailed -> use survival function / isf with p/2
    sigma_plus = stats.norm.isf(pval_plus / 2.0)
    sigma_cross = stats.norm.isf(pval_cross / 2.0)

    print(f"[g_plus]  chi2 = {chi2_plus:.2f}, dof = {dof}, p = {pval_plus:.2e}, ~ {sigma_plus:.2f}σ")
    print(f"[g_cross] chi2 = {chi2_cross:.2f}, dof = {dof}, p = {pval_cross:.2e}, ~ {sigma_cross:.2f}σ")

    # --- Plots ---
    # Noise only
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

    # Signal vs noise-subtracted
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

    # Noise-subtracted only
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

    # Noise vs noise-subtracted
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

    # Covariance heatmaps
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
# Run both normal and flipG1 cases
# ==============================================================#
cases = {
    "normal": ("", os.path.join(filament_dir, f"shear_p{final_percentile:02d}.csv")),
    "flipG1": ("_flipG1", os.path.join(filament_dir, f"shear_p{final_percentile:02d}_flipG1.csv")),
}

for case, (suffix, shear_csv) in cases.items():
    if not os.path.exists(shear_csv):
        print(f"[Warning] Missing shear file for {case} case: {shear_csv}")
        continue

    noise_files = [
        os.path.join(noise_shear_dir, f"shear_noise_p{final_percentile:02d}_{i:02d}{suffix}.csv")
        for i in range(num_realizations)
    ]
    run_analysis(case, shear_csv, noise_files, suffix)
