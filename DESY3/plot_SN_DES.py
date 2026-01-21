import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Global plotting style
# ============================================================
plt.rcParams.update({
    "figure.figsize": (8, 6.8),
    "figure.dpi": 100,

    "axes.linewidth": 1.6,
    "axes.labelsize": 15,
    "axes.titlesize": 15,

    # Major ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,

    # Minor ticks
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,

    "font.family": "serif",

    "legend.frameon": False,
    "legend.fontsize": 12,

    "savefig.bbox": "tight",
})


def inv_cov(cov, eps=1e-12):
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        print("[WARN] Covariance singular -> adding diagonal regularization.")
        cov_reg = cov + np.eye(cov.shape[0]) * eps
        return np.linalg.inv(cov_reg)


def plot_1d(arcmin_centers, y, yerr, ylabel, outpath, marker="o", label=None):
    plt.figure(figsize=(7.5, 5.2))
    plt.errorbar(
        arcmin_centers, y, yerr=yerr,
        fmt=f"{marker}-", capsize=4, elinewidth=1.2, markersize=4,
        label=label
    )
    plt.xscale("log")
    plt.xlabel("Separation [arcmin]")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    if label is not None:
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def run_analysis(case_label, shear_csv, noise_files, plot_dir, plot_suffix=""):
    print(f"\n=== Running analysis for {case_label} ===")
    os.makedirs(plot_dir, exist_ok=True)

    # --- Load signal ---
    signal_data = np.loadtxt(shear_csv, delimiter=",", skiprows=1)
    bin_center_rad = signal_data[:, 0]
    arcmin_centers = np.degrees(bin_center_rad) * 60.0

    g_plus_signal = signal_data[:, 2]
    g_cross_signal = signal_data[:, 3]

    # --- Load noise realizations ---
    all_g_plus_noise = []
    all_g_cross_noise = []

    used_noise = 0
    for nf in noise_files:
        if not os.path.exists(nf):
            continue
        data = np.loadtxt(nf, delimiter=",", skiprows=1)
        all_g_plus_noise.append(data[:, 2])
        all_g_cross_noise.append(data[:, 3])
        used_noise += 1

    if used_noise == 0:
        raise RuntimeError(f"No noise realizations found for {case_label}.")

    all_g_plus_noise = np.array(all_g_plus_noise)
    all_g_cross_noise = np.array(all_g_cross_noise)

    print(f"[INFO] Using {used_noise} noise realizations.")

    # --- Noise mean/std ---
    g_plus_noise_mean = np.mean(all_g_plus_noise, axis=0)
    g_cross_noise_mean = np.mean(all_g_cross_noise, axis=0)

    g_plus_noise_std = np.std(all_g_plus_noise, axis=0, ddof=1)
    g_cross_noise_std = np.std(all_g_cross_noise, axis=0, ddof=1)

    # --- Subtract mean noise ---
    g_plus_sub = g_plus_signal - g_plus_noise_mean
    g_cross_sub = g_cross_signal - g_cross_noise_mean

    # --- Covariance matrices from noise realizations (for sigma) ---
    cov_plus = np.cov(all_g_plus_noise, rowvar=False, ddof=1)
    cov_cross = np.cov(all_g_cross_noise, rowvar=False, ddof=1)

    cov_plus_inv = inv_cov(cov_plus)
    cov_cross_inv = inv_cov(cov_cross)

    # --- Chi-square against zero model (KEEP) ---
    d_plus = g_plus_sub
    d_cross = g_cross_sub
    dof = len(d_plus)

    chi2_plus = float(d_plus @ cov_plus_inv @ d_plus)
    chi2_cross = float(d_cross @ cov_cross_inv @ d_cross)

    pval_plus = 1 - stats.chi2.cdf(chi2_plus, dof)
    pval_cross = 1 - stats.chi2.cdf(chi2_cross, dof)

    sigma_plus = stats.norm.isf(pval_plus / 2.0) if pval_plus > 0 else np.inf
    sigma_cross = stats.norm.isf(pval_cross / 2.0) if pval_cross > 0 else np.inf

    print(f"[g_plus]  chi2 = {chi2_plus:.2f}, dof = {dof}, p = {pval_plus:.3e}, ~ {sigma_plus:.2f}σ")
    print(f"[g_cross] chi2 = {chi2_cross:.2f}, dof = {dof}, p = {pval_cross:.3e}, ~ {sigma_cross:.2f}σ")

    # ==========================================================
    # 4 separate plots (requested)
    # ==========================================================

    # 1) mean noise gamma+
    plot_1d(
        arcmin_centers,
        g_plus_noise_mean,
        g_plus_noise_std,
        ylabel=r"$\langle \gamma_+^{\rm rand} \rangle$",
        outpath=os.path.join(plot_dir, f"noise_mean_gplus{plot_suffix}.png"),
        marker="o",
    )

    # 2) mean noise gamma_x
    plot_1d(
        arcmin_centers,
        g_cross_noise_mean,
        g_cross_noise_std,
        ylabel=r"$\langle \gamma_{\times}^{\rm rand} \rangle$",
        outpath=os.path.join(plot_dir, f"noise_mean_gcross{plot_suffix}.png"),
        marker="s",
    )

    # 3) subtracted gamma+
    plot_1d(
        arcmin_centers,
        g_plus_sub,
        g_plus_noise_std,
        ylabel=r"$\gamma_+ - \langle \gamma_+^{\rm rand} \rangle$",
        outpath=os.path.join(plot_dir, f"signal_minus_noise_gplus{plot_suffix}.png"),
        marker="o",
    )

    # 4) subtracted gamma_x
    plot_1d(
        arcmin_centers,
        g_cross_sub,
        g_cross_noise_std,
        ylabel=r"$\gamma_{\times} - \langle \gamma_{\times}^{\rm rand} \rangle$",
        outpath=os.path.join(plot_dir, f"signal_minus_noise_gcross{plot_suffix}.png"),
        marker="s",
    )

    print(f"[DONE] Plots saved in: {plot_dir}")


if __name__ == "__main__":
    # ------------------------------------------------------------
    # KEEP directory logic exactly as in your script
    # ------------------------------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))

    shear_dir = os.path.join(current_dir, "shear")

    # plot directory (unchanged)
    plot_dir = os.path.join(shear_dir, "plots", "p15")
    os.makedirs(plot_dir, exist_ok=True)

    shear_csv = os.path.join(shear_dir, "shear_p15.csv")

    # if your noise files are named differently, only change this glob pattern
    noise_files = sorted(glob.glob(os.path.join(shear_dir, "shear_random_p15_*.csv")))

    if not os.path.exists(shear_csv):
        raise FileNotFoundError(f"Missing signal shear file: {shear_csv}")

    if len(noise_files) == 0:
        raise FileNotFoundError(f"No noise files found in: {shear_dir}")

    run_analysis(
        case_label="DESY3 p15",
        shear_csv=shear_csv,
        noise_files=noise_files,
        plot_dir=plot_dir,
        plot_suffix="_p15",
    )
