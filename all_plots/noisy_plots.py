import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# Path setup 

current_dir = os.path.dirname(os.path.abspath(__file__))      
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))  

os.chdir(current_dir)



#shear directories
DESY3_shear_dir = os.path.join(parent_dir, "DESY3/shear_flipg2")  
DESY3_shear_csv = os.path.join(DESY3_shear_dir, "shear_p15_flipg2.csv")

sim_shear_dir = os.path.join(parent_dir, "fiducial_sim/shear/DES_fiducial_sim/band_0.1_mesh_2/run_1_p15") 
sim_shear_csv = os.path.join(sim_shear_dir, "signal_shear.csv")

# noise_files 
DESY3_noise_files = sorted(glob.glob(os.path.join(DESY3_shear_dir, "shear_random_p15_*.csv")))  
sim_noise_files = sorted(glob.glob(os.path.join(sim_shear_dir, "random_rotations/shear_random_*.csv")))

# plot directory
DESY3_plot_dir = os.path.join(current_dir, "DESY3_plots")
sim_plot_dir = os.path.join(current_dir, "sim_plots")
os.makedirs(DESY3_plot_dir, exist_ok=True)
os.makedirs(sim_plot_dir, exist_ok=True)

# Global plotting style

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


# Cov regularization 
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


def run_analysis(case_label, shear_csv, noise_files, plot_dir):
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


    all_g_plus_noise = np.array(all_g_plus_noise)
    all_g_cross_noise = np.array(all_g_cross_noise)


    # --- Noise mean/std ---
    g_plus_noise_mean = np.mean(all_g_plus_noise, axis=0)
    g_cross_noise_mean = np.mean(all_g_cross_noise, axis=0)

    g_plus_noise_std = np.std(all_g_plus_noise, axis=0, ddof=1)
    g_cross_noise_std = np.std(all_g_cross_noise, axis=0, ddof=1)


    # --- Covariance matrices from noise realizations ---
    cov_plus = np.cov(all_g_plus_noise, rowvar=False, ddof=1)
    cov_cross = np.cov(all_g_cross_noise, rowvar=False, ddof=1)

    cov_plus_inv = inv_cov(cov_plus)
    cov_cross_inv = inv_cov(cov_cross)

    # --- Hartlap factor ---
    N = used_noise  # number of noise realizations 
    p = cov_plus.shape[0]
    hartlap = (N - p - 2) / (N - 1)  # Hartlap factor
    # Apply correction
    cov_plus_inv = hartlap * cov_plus_inv  
    cov_cross_inv = hartlap * cov_cross_inv  

    # --- Chi-square ---
    d_plus = g_plus_signal      
    d_cross = g_cross_signal    
    dof = len(d_plus)

    chi2_plus = float(d_plus @ cov_plus_inv @ d_plus)
    chi2_cross = float(d_cross @ cov_cross_inv @ d_cross)

    pval_plus = 1 - stats.chi2.cdf(chi2_plus, dof)
    pval_cross = 1 - stats.chi2.cdf(chi2_cross, dof)

    sigma_plus = stats.norm.isf(pval_plus / 2.0) if pval_plus > 0 else np.inf
    sigma_cross = stats.norm.isf(pval_cross / 2.0) if pval_cross > 0 else np.inf

    print(f"[g_plus]  chi2 = {chi2_plus:.2f}, dof = {dof}, p = {pval_plus:.3e}, ~ {sigma_plus:.2f}σ")
    print(f"[g_cross] chi2 = {chi2_cross:.2f}, dof = {dof}, p = {pval_cross:.3e}, ~ {sigma_cross:.2f}σ")

    
    # --- test sigma computation ---
    def chi2_to_sigma(delta_chi2, dof=1):
        """
        Convert Delta Chi-squared to Gaussian Sigma 
        """
        p_val = stats.chi2.cdf(delta_chi2, dof)
        sigma = stats.norm.ppf(p_val)
        return sigma

    print(f"[g_plus]  sigma_test = {chi2_to_sigma(chi2_plus, dof=dof):.2f}σ")
    print(f"[g_cross] sigma_test = {chi2_to_sigma(chi2_cross, dof=dof):.2f}σ")

    
    # -------------- plots -----------------------------
    

    # 1) mean noise gamma+
    plot_1d(
        arcmin_centers,
        g_plus_noise_mean,
        g_plus_noise_std,
        ylabel=r"$\langle \gamma_+^{\rm rand} \rangle$",
        outpath=os.path.join(plot_dir, f"noise_mean_gplus.png"),
        marker="o",
    )

    # 2) mean noise gamma_x
    plot_1d(
        arcmin_centers,
        g_cross_noise_mean,
        g_cross_noise_std,
        ylabel=r"$\langle \gamma_{\times}^{\rm rand} \rangle$",
        outpath=os.path.join(plot_dir, f"noise_mean_gcross.png"),
        marker="s",
    )

    # 3) signal gamma+
    plot_1d(
        arcmin_centers,
        g_plus_signal,              
        g_plus_noise_std,
        ylabel=r"$\gamma_+$",       
        outpath=os.path.join(plot_dir, f"signal_gplus.png"),
        marker="o",
    )
    
    # 4) signal gamma_x
    plot_1d(
        arcmin_centers,
        g_cross_signal,             
        g_cross_noise_std,
        ylabel=r"$\gamma_{\times}$",
        outpath=os.path.join(plot_dir, f"signal_gcross.png"),
        marker="s",
    )

    print(f"Plots saved in: {plot_dir}")


if __name__ == "__main__":

    run_analysis(
        case_label="DESY3",
        shear_csv=DESY3_shear_csv,
        noise_files=DESY3_noise_files,
        plot_dir=DESY3_plot_dir,
    )
    
    run_analysis(
        case_label="Simulation",
        shear_csv=sim_shear_csv,
        noise_files=sim_noise_files,
        plot_dir=sim_plot_dir,
    )
