import os
import glob
import numpy as np
import matplotlib.pyplot as plt

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
        try:
            data = np.loadtxt(nf, delimiter=",", skiprows=1)
        except Exception as e:
            print(f"[WARN] Could not read {nf}: {e}")
            continue

        all_g_plus_noise.append(data[:, 2])
        all_g_cross_noise.append(data[:, 3])
        used_noise += 1

    if used_noise == 0:
        raise RuntimeError(f"No readable noise realizations found for {case_label}.")

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

    # Plot formatting
    eb_kwargs = dict(fmt="o-", capsize=4, elinewidth=1.2, markersize=4)

    # ==========================================================
    # 1) Noise mean ± std: gamma+
    # ==========================================================
    plt.figure(figsize=(7.5, 5.2))
    plt.errorbar(arcmin_centers, g_plus_noise_mean, yerr=g_plus_noise_std,
                 label=r"$\langle \gamma_+^{\mathrm{rand}} \rangle$", **eb_kwargs)
    plt.xscale("log")
    plt.xlabel("Separation [arcmin]")
    plt.ylabel(r"$\gamma_+$")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"noise_mean_gplus{plot_suffix}.png"), dpi=200)
    plt.close()

    # ==========================================================
    # 2) Noise mean ± std: gamma_x
    # ==========================================================
    plt.figure(figsize=(7.5, 5.2))
    plt.errorbar(arcmin_centers, g_cross_noise_mean, yerr=g_cross_noise_std,
                 label=r"$\langle \gamma_{\times}^{\mathrm{rand}} \rangle$", **eb_kwargs)
    plt.xscale("log")
    plt.xlabel("Separation [arcmin]")
    plt.ylabel(r"$\gamma_{\times}$")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"noise_mean_gcross{plot_suffix}.png"), dpi=200)
    plt.close()

    # ==========================================================
    # 3) Noise-subtracted: gamma+
    # ==========================================================
    plt.figure(figsize=(7.5, 5.2))
    plt.errorbar(arcmin_centers, g_plus_sub, yerr=g_plus_noise_std,
                 label=r"$\gamma_+ - \langle \gamma_+^{\mathrm{rand}} \rangle$", **eb_kwargs)
    plt.xscale("log")
    plt.xlabel("Separation [arcmin]")
    plt.ylabel(r"$\gamma_+$")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"signal_minus_noise_gplus{plot_suffix}.png"), dpi=200)
    plt.close()

    # ==========================================================
    # 4) Noise-subtracted: gamma_x
    # ==========================================================
    plt.figure(figsize=(7.5, 5.2))
    plt.errorbar(arcmin_centers, g_cross_sub, yerr=g_cross_noise_std,
                 label=r"$\gamma_{\times} - \langle \gamma_{\times}^{\mathrm{rand}} \rangle$", **eb_kwargs)
    plt.xscale("log")
    plt.xlabel("Separation [arcmin]")
    plt.ylabel(r"$\gamma_{\times}$")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"signal_minus_noise_gcross{plot_suffix}.png"), dpi=200)
    plt.close()

    print(f"[DONE] 4 plots saved in: {plot_dir}")


if __name__ == "__main__":
    # base_dir = directory containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    shear_csv = os.path.join(base_dir, "shear/run_1/band_0.1/mesh_2/run_1_p15_signal_shear.csv")
    noise_files = sorted(glob.glob(
        os.path.join(base_dir, "shear/run_1/band_0.1/mesh_2/random_rotations", "shear_random_*.csv")
    ))

    plot_dir = os.path.join(base_dir, "plots", "p15")

    run_analysis(
        case_label="run_1 band_0.1 mesh_2 p15",
        shear_csv=shear_csv,
        noise_files=noise_files,
        plot_dir=plot_dir,
        plot_suffix="_run1_band0p1_mesh2_p15",
    )
