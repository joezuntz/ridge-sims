import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

# Path setup --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))      
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))  


os.chdir(current_dir)

#parameters --------------------------------------------------

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


# Output directory

output = os.path.join(current_dir, "parameter_plots")
os.makedirs(output, exist_ok=True)
print(f"\nAll plots will be saved to: {output}\n")



dist   = "Weighted_Real_Distance"
gplus = "Weighted_g_plus"
gcross = "Weighted_g_cross"



# loading function

def load_shear_files(data_dir, tag):
    """
    Loads CSVs matching:
        shear_<tag>_<value>.csv
    """
    prefix = f"shear_{tag}_"
    out = []

    for f in os.listdir(data_dir):
        if not (f.startswith(prefix) and f.endswith(".csv")):
            continue

        value = float(f[len(prefix):-4])  
        df = pd.read_csv(os.path.join(data_dir, f))
        out.append((value, df))
        print(f"Loaded: {os.path.join(data_dir, f)}")

    out.sort(key=lambda x: x[0])
    return out



# plot function

def plot_shear_family(shear_data, param_label, out_prefix):
    """
    shear_data: list of (param_value, df)
    Saves:
      <out_prefix>_gplus.pdf
      <out_prefix>_gcross.pdf
    """

    fig_gplus, ax_gplus = plt.subplots()
    fig_gx, ax_gx       = plt.subplots()

    # Colormap
    vals = [v for v, _ in shear_data]
    norm = colors.Normalize(vmin=min(vals), vmax=max(vals))
    cmap = cm.viridis
    sm   = cm.ScalarMappable(norm=norm, cmap=cmap)

    for v, df in shear_data:
        theta_arcmin = np.degrees(df[dist].values) * 60.0
        color = cmap(norm(v))

        ax_gplus.plot(theta_arcmin, df[gplus].values, color=color, alpha=0.85)
        ax_gx.plot(theta_arcmin, df[gcross].values, color=color, alpha=0.85)

    # gamma + 
    ax_gplus.set_xscale("log")
    ax_gplus.set_xlabel(r"$\theta$ [arcmin]")
    ax_gplus.set_ylabel(r"$\gamma_{+}$")
    ax_gplus.set_title(r"$\gamma_{+}$ vs $\theta$")
    cbar_gplus = fig_gplus.colorbar(sm, ax=ax_gplus, pad=0.02)
    cbar_gplus.set_label(param_label)

    # gamma x 
    ax_gx.set_xscale("log")
    ax_gx.set_xlabel(r"$\theta$ [arcmin]")
    ax_gx.set_ylabel(r"$\gamma_{\times}$")
    ax_gx.set_title(r"$\gamma_{\times}$ vs $\theta$")
    cbar_gx = fig_gx.colorbar(sm, ax=ax_gx, pad=0.02)
    cbar_gx.set_label(param_label)

    # save 
    out_gplus = os.path.join(output, f"{out_prefix}_gplus.pdf")
    out_gx    = os.path.join(output, f"{out_prefix}_gcross.pdf")

    fig_gplus.savefig(out_gplus)
    fig_gx.savefig(out_gx)

    plt.close(fig_gplus)
    plt.close(fig_gx)

    print("\nSaved:")
    print(" →", out_gplus)
    print(" →", out_gx, "\n")



# Specific plots

def plot_shear_meshsizes():
    mesh_dir = os.path.join(parent_dir, "hyperparameter_test/parameter_test/shear_vs_meshsize")
    shear_data = load_shear_files(mesh_dir, tag="mesh")
    plot_shear_family(shear_data, param_label="Mesh size", out_prefix="mesh_sizes")

def plot_shear_bandwidths():
    band_dir = os.path.join(parent_dir, "hyperparameter_test/parameter_test/shear_vs_bandwidth")
    shear_data = load_shear_files(band_dir, tag="band")
    plot_shear_family(shear_data, param_label="Bandwidth", out_prefix="bandwidths")

def plot_shear_fp():
    fp_dir = os.path.join(parent_dir, "hyperparameter_test/parameter_test/shear_vs_fp")
    shear_data = load_shear_files(fp_dir, tag="fp")
    plot_shear_family(shear_data, param_label="Density threshold", out_prefix="threshold")



# ============================================================
if __name__ == "__main__":
    plot_shear_meshsizes()
    plot_shear_bandwidths()
    plot_shear_fp()
