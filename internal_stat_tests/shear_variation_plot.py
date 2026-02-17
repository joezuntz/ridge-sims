import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==============================================================
# PATH SETUP 
# ==============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# ==============================================================
# Global plotting style 
# ==============================================================
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
    "legend.fontsize": 12,

    "savefig.bbox": "tight",
})

#Functions

def inv_cov(cov, eps=1e-12):
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        print("[WARN] Covariance singular -> adding diagonal regularization.")
        cov_reg = cov + np.eye(cov.shape[0]) * eps
        return np.linalg.inv(cov_reg)

def load_shear_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    theta_rad = data[:, 0]
    arcmin = np.degrees(theta_rad) * 60.0
    g_plus = data[:, 2]
    g_cross = data[:, 3]
    return arcmin, g_plus, g_cross

def plot_1d(arcmin_centers, y, yerr, ylabel, marker="o", label=None):
    plt.errorbar(
        arcmin_centers, y, yerr=yerr,
        fmt=marker, capsize=4, elinewidth=1.2, markersize=4,
        label=label
    )
    plt.xscale("log")
    plt.xlabel("Separation [arcmin]")
    plt.ylabel(ylabel)
    plt.xlim(1, 60)

def chi2_to_sigma(chi2, dof):
    # two-sided gaussian-equivalent sigma from chi2 p-value
    p = stats.chi2.sf(chi2, dof)
    return stats.norm.isf(p / 2.0) if p > 0 else np.inf

# Analysis
def run_internal_scatter_test(case_label, signal_csv, ensemble_csvs, xmove=1.0):
    print(f"\n=== Internal scatter test: {case_label} ===")

    # --- Signal (run 00) ---
    arcmin, gplus_sig, gcross_sig = load_shear_csv(signal_csv)

    # --- Ensemble (runs 01..49) ---
    gplus_list = []
    gcross_list = []
    used = 0

    for f in ensemble_csvs:
        if not os.path.exists(f):
            continue
        arcmin_i, gp_i, gx_i = load_shear_csv(f)

        # bin mismatch error
        if len(arcmin_i) != len(arcmin) or np.max(np.abs(arcmin_i - arcmin)) > 1e-10:
            raise ValueError(f"Binning mismatch between {signal_csv} and {f}")

        gplus_list.append(gp_i)
        gcross_list.append(gx_i)
        used += 1

    gplus_arr = np.array(gplus_list)
    gcross_arr = np.array(gcross_list)



    # --- Ensemble mean/std (per bin) ---
    gplus_mean = np.mean(gplus_arr, axis=0)
    gcross_mean = np.mean(gcross_arr, axis=0)

    gplus_std = np.std(gplus_arr, axis=0, ddof=1)
    gcross_std = np.std(gcross_arr, axis=0, ddof=1)

    # --- Covariance from ensemble ---
    cov_plus = np.cov(gplus_arr, rowvar=False, ddof=1)
    cov_cross = np.cov(gcross_arr, rowvar=False, ddof=1)

    cov_plus_inv = inv_cov(cov_plus)
    cov_cross_inv = inv_cov(cov_cross)

    # --- Hartlap correction (optional; keep it because you asked for p-values "and stuff") ---
    N = used
    p = cov_plus.shape[0]
    if N > p + 2:
        hartlap = (N - p - 2) / (N - 1)
        cov_plus_inv  = hartlap * cov_plus_inv
        cov_cross_inv = hartlap * cov_cross_inv
    else:
        print(f"[WARN] Hartlap invalid (N={N}, p={p}) -> skipping Hartlap correction.")

    # --- Reproducibility residuals: compare run00 to ensemble mean ---
    d_plus  = gplus_sig  - gplus_mean
    d_cross = gcross_sig - gcross_mean
    dof = len(d_plus)

    chi2_plus  = float(d_plus  @ cov_plus_inv  @ d_plus)
    chi2_cross = float(d_cross @ cov_cross_inv @ d_cross)

    pval_plus  = stats.chi2.sf(chi2_plus, dof)
    pval_cross = stats.chi2.sf(chi2_cross, dof)

    sigma_plus  = chi2_to_sigma(chi2_plus, dof)
    sigma_cross = chi2_to_sigma(chi2_cross, dof)

    print(f"[g_plus]  chi2 = {chi2_plus:.3f}, dof = {dof}, chi2/dof = {chi2_plus/dof:.3f}, p = {pval_plus:.3e}, ~ {sigma_plus:.2f}σ")
    print(f"[g_cross] chi2 = {chi2_cross:.3f}, dof = {dof}, chi2/dof = {chi2_cross/dof:.3f}, p = {pval_cross:.3e}, ~ {sigma_cross:.2f}σ")

    # --- Stability diagnostics ---
    rel_scatter = np.max(np.abs(gplus_std / np.where(gplus_mean != 0, gplus_mean, np.nan)))
    print(f"[diag] used ensemble files = {used}")
    print(f"[diag] max |std/mean| for g_plus (ignoring mean=0 bins) = {rel_scatter:.3e}")

    # --- Plot: run00 with std error bars, plus ensemble mean curve ---
    plot_1d(
        arcmin * xmove,
        gplus_sig / 1e-3,
        gplus_std / 1e-3,
        ylabel=r"$\gamma_+$ / $10^{-3}$",
        marker="o"
    )
    plt.plot(
        arcmin * xmove,
        gplus_mean / 1e-3,
        lw=1.6
    )

    return {
        "used": used,
        "arcmin": arcmin,
        "gplus_sig": gplus_sig,
        "gplus_mean": gplus_mean,
        "gplus_std": gplus_std,
        "chi2_plus": chi2_plus,
        "pval_plus": pval_plus,
        "sigma_plus": sigma_plus,
        "chi2_cross": chi2_cross,
        "pval_cross": pval_cross,
        "sigma_cross": sigma_cross,
    }

# ==============================================================
# Main
# ==============================================================
if __name__ == "__main__":

    repeats_dir = os.path.join("shear_stat_test", "shear_repeats")

    signal_csv = os.path.join(repeats_dir, "shear_00.csv")
    ensemble_csvs = [os.path.join(repeats_dir, f"shear_{i:02d}.csv") for i in range(1, 50)]

    outdir = os.path.join(repeats_dir, "plots_internal_scatter")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "internal_scatter_gplus.pdf")

    plt.figure(figsize=(6, 4.5))

    run_internal_scatter_test(
        case_label="Shear repeats",
        signal_csv=signal_csv,
        ensemble_csvs=ensemble_csvs,
        xmove=1.0
    )

    plt.tight_layout()
    plt.axhline(0, color="black", ls="-", lw=1.0)
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"Saved: {outpath}")
