import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# ----------------------------
# Path setup
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))
os.chdir(current_dir)

# ----------------------------
# Plot style 
# ----------------------------
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
    "legend.fontsize": 11,
    "savefig.bbox": "tight",
})

def load_shear_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    bin_center_rad = data[:, 0]
    arcmin_centers = np.degrees(bin_center_rad) * 60.0
    g_plus  = data[:, 2]
    g_cross = data[:, 3]
    return arcmin_centers, g_plus, g_cross

def load_noise_realizations(noise_dir):
    files = sorted(glob.glob(os.path.join(noise_dir, "*.csv")))
    if len(files) == 0:
        raise FileNotFoundError(f"No noise CSVs found in: {noise_dir}")

    gp_list, gx_list = [], []
    for f in files:
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        gp_list.append(d[:, 2])
        gx_list.append(d[:, 3])

    gp = np.asarray(gp_list)
    gx = np.asarray(gx_list)
    return gp, gx, len(files)

def stacked_threshold_plot(
    root_dir,
    run_id=1,
    thresholds=(10, 15, 25, 40, 55, 70),
    outdir="density_threshold_test_plot",
    outfile="gplus_vs_thresholds.pdf",
    which="gplus",          # "gplus" or "gcross"
    scale=1e-3,             # plot scale
    x_jitter_frac=0.010,   
):
    os.makedirs(outdir, exist_ok=True)

    norm = Normalize(vmin=min(thresholds), vmax=max(thresholds))
    cmap = cm.get_cmap("coolwarm")

    plt.figure(figsize=(6.6, 4.8))
    ax = plt.gca()

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for i, thr in enumerate(thresholds):
        shear_csv = os.path.join(root_dir, f"run_{run_id}_shear_fp_{thr}.csv")
        noise_dir = os.path.join(root_dir, f"run_{run_id}_random_shear_p{thr}")

        x_arcmin, gp_sig, gx_sig = load_shear_csv(shear_csv)
        gp_noise, gx_noise, Nrand = load_noise_realizations(noise_dir)

        gp_std = np.std(gp_noise, axis=0, ddof=1)
        gx_std = np.std(gx_noise, axis=0, ddof=1)

        color = cmap(norm(thr))

        xmove = 1.0 + (i - 0.5 * (len(thresholds) - 1)) * x_jitter_frac

        if which == "gplus":
            y = gp_sig / scale
            yerr = gp_std / scale
            ylabel = r"$\gamma_+$ / $10^{-3}$"
        elif which == "gcross":
            y = gx_sig / scale
            yerr = gx_std / scale
            ylabel = r"$\gamma_{\times}$ / $10^{-3}$"
        else:
            raise ValueError("which must be 'gplus' or 'gcross'")

        ax.errorbar(
            x_arcmin * xmove,
            y,
            yerr=yerr,
            fmt="o",
            capsize=3,
            elinewidth=1.1,
            markersize=3.5,
            color=color,
            alpha=0.95,
        )

    ax.set_xscale("log")
    ax.set_xlim(1, 60)
    ax.set_xlabel("Separation [arcmin]")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", lw=1.0)

    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Density thresholds")


    outpath = os.path.join(outdir, outfile)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"Wrote: {outpath}")

if __name__ == "__main__":
    root = os.path.join(
                        parent_dir,
                        "fiducial_sim",
                        "density_threshold_test",
                        "shear_vs_threshold",
                    )
    stacked_threshold_plot(
        root_dir=root,
        run_id=1,
        thresholds=(10, 15, 25, 40, 55, 70),
        outdir=os.path.join(current_dir, "density_threshold_test_plots"),
        outfile="stacked_gplus_thresholds2.pdf",
        which="gplus",
        scale=1e-3,
        x_jitter_frac=0.010,
    )