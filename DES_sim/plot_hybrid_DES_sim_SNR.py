import os
import glob
import re
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

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,

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




_noise_re = re.compile(r"shear_random_(\d+)\.csv$")

def natural_noise_sort_key(path):
    m = _noise_re.search(os.path.basename(path))
    return int(m.group(1)) if m else 10**18


def compute_snr_cov(d, cov, ridge=1e-12):
    """
    SNR_cov = sqrt(d^T C^{-1} d)
    
    """
    cov = np.asarray(cov, float)
    d = np.asarray(d, float)

    # regularization
    cov_reg = cov.copy()
    cov_reg.flat[:: cov_reg.shape[0] + 1] += ridge * (np.trace(cov_reg) / cov_reg.shape[0])

    inv = np.linalg.pinv(cov_reg)
    chi2 = float(d.T @ inv @ d)
    return np.sqrt(max(chi2, 0.0)), chi2


def run_analysis(case_label, shear_csv, noise_glob, plot_dir, plot_suffix=""):
    os.makedirs(plot_dir, exist_ok=True)

    # -------------------------
    # Load signal
    # -------------------------
    sig = np.loadtxt(shear_csv, delimiter=",", skiprows=1)
    bin_center_rad = sig[:, 0]
    arcmin = np.degrees(bin_center_rad) * 60.0
    gplus_sig = sig[:, 2]
    gcross_sig = sig[:, 3]

    # -------------------------
    # Load noise 
    # -------------------------
    noise_files = sorted(glob.glob(noise_glob), key=natural_noise_sort_key)

    gplus_noise = []
    gcross_noise = []
    for nf in noise_files:
        dat = np.loadtxt(nf, delimiter=",", skiprows=1)
        gplus_noise.append(dat[:, 2])
        gcross_noise.append(dat[:, 3])

    gplus_noise = np.array(gplus_noise)   # (Nreal, Nbin)
    gcross_noise = np.array(gcross_noise)

    # -------------------------
    # Noise stats
    # -------------------------
    gplus_mean = gplus_noise.mean(axis=0)
    gcross_mean = gcross_noise.mean(axis=0)

    gplus_std = gplus_noise.std(axis=0, ddof=1)
    gcross_std = gcross_noise.std(axis=0, ddof=1)

    cov_plus = np.cov(gplus_noise, rowvar=False, ddof=1)
    cov_cross = np.cov(gcross_noise, rowvar=False, ddof=1)

    # -------------------------
    # Subtracted vectors
    # -------------------------
    d_plus = gplus_sig - gplus_mean
    d_cross = gcross_sig - gcross_mean

    # -------------------------
    # Diagonal S/N
    # -------------------------
    snr_diag_plus = np.sqrt(np.sum((d_plus / gplus_std) ** 2))
    snr_diag_cross = np.sqrt(np.sum((d_cross / gcross_std) ** 2))

    # -------------------------
    # Full-cov S/N and chi2
    # -------------------------
    snr_cov_plus, chi2_plus = compute_snr_cov(d_plus, cov_plus)
    snr_cov_cross, chi2_cross = compute_snr_cov(d_cross, cov_cross)

    dof = len(d_plus)

    # -------------------------
    # p-value and "sigma" (two-sided)
    # -------------------------
    p_plus = 1.0 - stats.chi2.cdf(chi2_plus, dof)
    p_cross = 1.0 - stats.chi2.cdf(chi2_cross, dof)

    sigma_plus = stats.norm.isf(p_plus / 2.0)
    sigma_cross = stats.norm.isf(p_cross / 2.0)

    # -------------------------
    # Save summary
    # -------------------------
    out_txt = os.path.join(plot_dir, f"sigma_detection{plot_suffix}.txt")
    with open(out_txt, "w") as f:
        f.write(f"Case: {case_label}\n")
        f.write(f"Signal: {shear_csv}\n")
        f.write(f"Noise glob: {noise_glob}\n")
        f.write(f"N_noise: {len(noise_files)}\n")
        f.write(f"dof: {dof}\n\n")

        f.write("gamma+\n")
        f.write(f"  snr_diag = {snr_diag_plus:.4f}\n")
        f.write(f"  snr_cov  = {snr_cov_plus:.4f}\n")
        f.write(f"  chi2     = {chi2_plus:.4f}\n")
        f.write(f"  p-value  = {p_plus:.4e}\n")
        f.write(f"  sigma    = {sigma_plus:.4f}\n\n")

        f.write("gamma_x\n")
        f.write(f"  snr_diag = {snr_diag_cross:.4f}\n")
        f.write(f"  snr_cov  = {snr_cov_cross:.4f}\n")
        f.write(f"  chi2     = {chi2_cross:.4f}\n")
        f.write(f"  p-value  = {p_cross:.4e}\n")
        f.write(f"  sigma    = {sigma_cross:.4f}\n")

    print("[DONE]", out_txt)
    print(f"[gamma+]  snr_cov={snr_cov_plus:.3f}  chi2={chi2_plus:.2f}  p={p_plus:.2e}  sigma~{sigma_plus:.2f}")
    print(f"[gammax]  snr_cov={snr_cov_cross:.3f}  chi2={chi2_cross:.2f}  p={p_cross:.2e}  sigma~{sigma_cross:.2f}")

    # -------------------------
    # Plots: noise-subtracted gamma+ and gamma_x
    # -------------------------
    eb_kwargs = dict(
        fmt="o-",
        capsize=4,
        elinewidth=1.2,
        markersize=4,
    )

    # gamma+
    plt.figure(figsize=(7.5, 5.2))
    plt.errorbar(
        arcmin,
        d_plus,
        yerr=gplus_std,
        **eb_kwargs
    )
    plt.xscale("log")
    plt.xlabel("Separation [arcmin]")
    plt.ylabel(r"$\gamma_+ - \langle \gamma_+^{\rm rand} \rangle$")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f"signal_minus_noise_gplus{plot_suffix}.png"),
        dpi=200
    )
    plt.close()

    # gamma_x
    plt.figure(figsize=(7.5, 5.2))
    plt.errorbar(
        arcmin,
        d_cross,
        yerr=gcross_std,
        **eb_kwargs
    )
    plt.xscale("log")
    plt.xlabel("Separation [arcmin]")
    plt.ylabel(r"$\gamma_{\times} - \langle \gamma_{\times}^{\rm rand} \rangle$")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f"signal_minus_noise_gcross{plot_suffix}.png"),
        dpi=200
    )
    plt.close()



if __name__ == "__main__":
    base_dir = os.getcwd()

    target_dir = os.path.join(
        base_dir,
        "shear_hybrid_DES",
        "run_1",
        "band_0.1",
        "mesh_2",
    )

    shear_csv = os.path.join(target_dir, "run_1_p15_signal_shear.csv")
    noise_glob = os.path.join(target_dir, "random_rotations", "shear_random_*.csv")

    plot_dir = os.path.join("shear_hybrid_DES", "plots_p15_hybrid")

    run_analysis(
        case_label="DES_sim shear_hybrid_DES run_1 band_0.1 mesh_2 p15",
        shear_csv=shear_csv,
        noise_glob=noise_glob,
        plot_dir=plot_dir,
        plot_suffix="_DES_run1_band0p1_mesh2_p15",
    )

