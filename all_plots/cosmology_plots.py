import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

# Path setup --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))      
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))  

os.chdir(os.path.join(parent_dir, "cosmo_sims"))

# parameters --------------------------------------------------

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

# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------
ROOT = "Cosmo_sim2_ridges"
CATEGORIES = ["Om_fixed", "S8", "S8_perp", "sigma8_fixed"]
RUNS = [f"run_{i}" for i in range(1, 11)]
P = 15

OUTPUT_ROOT = os.path.join(current_dir, "cosmo_plots2") 
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"\nAll plots will be saved to: {OUTPUT_ROOT}\n")


# ------------------------------------------------------------
# Load cosmology mapping
# ------------------------------------------------------------

mapping_path = "cosmo_run_mapping.csv"
mapping_df = pd.read_csv(mapping_path)

def get_param_label(category, run):
    row = mapping_df[(mapping_df.category == category) &
                     (mapping_df.run == run)]

    if len(row) == 0:
        return run

    row = row.iloc[0]

    if category == "Om_fixed":
        return rf"$\Omega_m = {row['Omega_m']:.3f}$"

    elif category == "S8":
        return rf"$S_8 = {row['S8']:.3f}$"

    elif category == "S8_perp":
        return rf"$S_8^\perp: \Omega_m = {row['Omega_m']:.3f}$"

    elif category == "sigma8_fixed":
        return rf"$\sigma_8 = {row['sigma8']:.3f}$"

    else:
        return ""

def get_param_value(category, run):
    """Numeric value used for colorbar (depends on what varies)."""
    row = mapping_df[(mapping_df.category == category) &
                     (mapping_df.run == run)]

    if len(row) == 0:
        return None

    row = row.iloc[0]

    if category == "sigma8_fixed":
        
        return row["Omega_m"]

    elif category == "Om_fixed":
        
        return row["sigma8"]

    elif category == "S8":
        return row["S8"]

    elif category == "S8_perp":
        return row["Omega_m"]

    else:
        return None



def load_shear_file(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

# --- fiducial values --------------
Omega_m_fid = 0.32
S8_fid = 0.78
sigma8_fid = S8_fid / np.sqrt(Omega_m_fid / 0.3)

def get_fiducial_value_for_category(category):
    # returns the fiducial "val" in the same variable used for colorbar 
    if category == "sigma8_fixed":
        return Omega_m_fid         
    elif category == "Om_fixed":
        return sigma8_fid           
    elif category == "S8":
        return S8_fid               
    elif category == "S8_perp":
        return Omega_m_fid          
    else:
        return None

# ------------------------------------------------------------
# Main plotting
# ------------------------------------------------------------
CATEGORY_TITLES = {
    "Om_fixed": r"$\sigma_8$",
    "sigma8_fixed": r"$\Omega_m$",
    "S8": r"$S_8$",
    "S8_perp": r"$S_8^{\perp}$      ",  # the extra spaces here stop the perp symbol from overlapping with colorbar
}


def plot_shear_all_categories():

    for cat in CATEGORIES:
        print(f"=== Category: {cat} ===")
        shear_list = []
        param_values = []

        for run in RUNS:

            csv_path = os.path.join(
                ROOT, cat, run,
                "band_0.1", "Ridges_final_p15", "shear",
                f"shear_p{P:02d}.csv"
            )

            df = load_shear_file(csv_path)
            if df is None:
                print(f"  [missing] {csv_path}")
                continue

            val = get_param_value(cat, run)
            if val is None:
                continue

            shear_list.append((df, val))
            param_values.append(val)

            print(f"  [loaded] {csv_path}")

        if len(shear_list) == 0:
            print(f"  No shear files for {cat}.")
            continue

        norm = Normalize(vmin=min(param_values), vmax=max(param_values))
        cmap = cm.get_cmap("coolwarm")

        fig_gplus, ax_gplus = plt.subplots(figsize=(6, 4.5))
        # fig_gx, ax_gx = plt.subplots(figsize=(8, 6))

        # identify fiducial
        fid_val = get_fiducial_value_for_category(cat)  
        vals_arr = np.array([v for _, v in shear_list], dtype=float)  

        atol = 1e-6  
        fid_mask = np.isclose(vals_arr, fid_val, atol=atol, rtol=0) if fid_val is not None else np.zeros_like(vals_arr, dtype=bool) 

        # nearest choice below tolerance
        if fid_val is not None and not np.any(fid_mask):  
            idx = int(np.argmin(np.abs(vals_arr - fid_val)))  
            fid_mask = np.zeros_like(vals_arr, dtype=bool)  
            fid_mask[idx] = True  

        rad_col = "Weighted_Real_Distance"
        gplus_col = "Weighted_g_plus"
        gcross_col = "Weighted_g_cross"
        gcross_style = 'dashed'



        # Plot non-fiducial first  
        for (df, val), is_fid in zip(shear_list, fid_mask):
            df[gplus_col] /= 1e-3
            df[gcross_col] /= 1e-3
            if is_fid: 
                continue  

            arcmin_centers = np.degrees(df[rad_col].values) * 60.0
            color = cmap(norm(val))

            ax_gplus.plot(arcmin_centers, df[gplus_col], alpha=0.6, color=color, lw=1.6) 
            ax_gplus.plot(arcmin_centers, df[gcross_col], alpha=0.6, color=color, lw=1.6, linestyle=gcross_style)  

        # # Plot fiducial last with outline + thicker stroke 
        # for (df, val), is_fid in zip(shear_list, fid_mask):  
        #     if not is_fid: 
        #         continue  

        #     arcmin_centers = np.degrees(df[rad_col].values) * 60.0
        #     fid_color = cmap(norm(val))

        #     ax_gplus.plot(arcmin_centers, df[gplus_col], color="k", lw=3.4, alpha=0.95, zorder=10)  
        #     ax_gplus.plot(arcmin_centers, df[gplus_col], color=fid_color, lw=2.4, alpha=1.0, zorder=11)  

        #     ax_gplus.plot(arcmin_centers, df[gcross_col], color="k", lw=3.4, alpha=0.95, zorder=10, linestyle=gcross_style)  
        #     ax_gplus.plot(arcmin_centers, df[gcross_col], color=fid_color, lw=2.4, alpha=1.0, zorder=11, linestyle=gcross_style)

        
        # Colorbar
        
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        title_cat = CATEGORY_TITLES.get(cat, cat)

        fig_gplus.colorbar(sm, ax=ax_gplus, pad=0.02, label=title_cat)
        # fig_gx.colorbar(sm, ax=ax_gx, pad=0.02)

        
        # g+
                
        ax_gplus.set_xscale("log")
        ax_gplus.set_ylim(-2, 1.5)
        # ax_gplus.set_xscale("log")
        # ax_gplus.set_title(title_cat + r": $\gamma_{+}$")
        ax_gplus.set_xlabel(r"$\theta [arcmin]$")
        ax_gplus.set_ylabel(r" $\gamma /  10^{-3}$")

        
        # g×
   
        # ax_gx.set_xscale("log")
        # ax_gx.set_title(title_cat + r": $\gamma_{\times}$")
        # ax_gx.set_xlabel(r"$\theta [arcmin]$")
        # ax_gx.set_ylabel(r"$\gamma_{times}$")

        # Add legend with proxy artists
        legend_elements = [
            Line2D([0], [0], color='black', lw=1.6, label=r'$\gamma_{+}$'),
            Line2D([0], [0], color='black', lw=1.6, linestyle='dashed', label=r'$\gamma_{\times}$')
        ]
        ax_gplus.legend(handles=legend_elements, loc='best')

        # Save 
        plt.tight_layout()
        fig_gplus.savefig(
            os.path.join(OUTPUT_ROOT, f"{cat}_gplus_all_runs.pdf"),
            bbox_inches="tight"
        )
        # fig_gx.savefig(
        #     os.path.join(OUTPUT_ROOT, f"{cat}_gcross_all_runs.pdf"),
        #     bbox_inches="tight"
        # )

        plt.close(fig_gplus)
        # plt.close(fig_gx)

        print(f" saved {cat}_gplus_all_runs.pdf")
        print(f" saved {cat}_gcross_all_runs.pdf\n")

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    plot_shear_all_categories()
