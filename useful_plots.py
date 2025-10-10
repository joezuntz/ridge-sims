import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from astropy.coordinates import SkyCoord
import astropy.units as u
from ridge_analysis_tools import *
## --- config ---
#base_dir = "example30_band0.4/8test"
#num_runs = 8
#final_percentile = 15
#output_dir = os.path.join(base_dir, "useful_plots")
#os.makedirs(output_dir, exist_ok=True)

#for run_id in range(1, num_runs + 1):
#    ridge_file = os.path.join(
#        base_dir, f"run_{run_id}", "ridges_filtered", f"ridges_p{final_percentile:02d}_filtered.h5"
#    )
#    if not os.path.exists(ridge_file):
#        print(f"[Run {run_id}] No file found: {ridge_file}")
#        continue

#    with h5py.File(ridge_file, "r") as f:
#        ridges = f["ridges"][:]

#    if ridges.size == 0:
#        print(f"[Run {run_id}] File exists but contains no ridges")
#        continue

#    ridge_dec = ridges[:, 0]
#    ridge_ra = ridges[:, 1]

#    plt.figure(figsize=(6, 4))
#    plt.scatter(ridge_ra, ridge_dec, s=1, alpha=0.6)
#    plt.xlabel("RA")
#    plt.ylabel("Dec")
#    plt.title(f"Run {run_id} – Filtered ridges (p{final_percentile})")
#    plt.tight_layout()
    
#    output_filename = f"run_{run_id}_ridges_p{final_percentile:02d}.png"
#    output_plot_path = os.path.join(output_dir, output_filename)

#    # Save the plot to the correct path
#    plt.savefig(output_plot_path, dpi=150)
#    plt.close()
#    print(f"Plot saved → {output_plot_path}")
    


## --- config ---
#base_dir = "example30_band0.4/8test"
#noise_dir = os.path.join(base_dir, "noise_data")
#output_dir = os.path.join(base_dir, "useful_plots")
#os.makedirs(output_dir, exist_ok=True)

## how many random files to inspect per run
#num_samples = 1
#runs_to_check = [1, 2, 3]

#for run_id in runs_to_check:
#    for _ in range(num_samples):
#        noise_idx = random.randint(0, 299)
#        noise_file = os.path.join(noise_dir, f"noise_r{run_id:02d}_n{noise_idx:03d}.h5")

#        if not os.path.exists(noise_file):
#            print(f"[Run {run_id}] File not found: {noise_file}")
#            continue

#        print(f"\n[Run {run_id}] Inspecting {noise_file}")
#        with h5py.File(noise_file, "r") as f:
#            keys = list(f.keys())
#            print("  Keys in file:", keys)

#            try:
#                dec = f["DEC"][:]
#                ra = f["RA"][:]
#                g1 = f["G1"][:]
#                g2 = f["G2"][:]
#            except KeyError as e:
#                print(f"  Missing expected dataset: {e}")
#                continue

#        print(f"  -> Catalog size: {len(ra)} objects")

#        # scatter plot RA/Dec
#        plt.figure(figsize=(6, 4))
#        plt.scatter(ra, dec, s=0.2, alpha=0.5)
#        plt.xlabel("RA")
#        plt.ylabel("Dec")
#        plt.title(f"Run {run_id} – Noise n{noise_idx:03d} (RA/Dec)")
#        plt.tight_layout()
#        out_png = os.path.join(output_dir, f"run{run_id}_noise{noise_idx:03d}_radec.png")
#        plt.savefig(out_png, dpi=150)
#        plt.close()
#        print(f"  -> RA/Dec plot saved to {out_png}")

#        # histogram of g1/g2
#        plt.figure(figsize=(6, 4))
#        plt.hist(g1, bins=100, alpha=0.6, label="G1")
#        plt.hist(g2, bins=100, alpha=0.6, label="G2")
#        plt.xlabel("Shear value")
#        plt.ylabel("Count")
#        plt.title(f"Run {run_id} – Noise n{noise_idx:03d} (G1/G2)")
#        plt.legend()
#        plt.tight_layout()
#        out_png = os.path.join(output_dir, f"run{run_id}_noise{noise_idx:03d}_g1g2.png")
#        plt.savefig(out_png, dpi=150)
#        plt.close()
#        print(f"  -> G1/G2 histogram saved to {out_png}")



#def plot_separate_backgrounds(base_sim_dir="lhc_run_sims_50", num_runs=3):
#    """
#    Plots the RA and DEC of background galaxies for each run separately.
#    """
#    base_dir = "example30_band0.4/8test"
#    output_dir = os.path.join(base_dir, "useful_plots")
#    os.makedirs(output_dir, exist_ok=True)
    
   
#    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10), sharex=True, sharey=True)
#    axes = axes.flatten() 
    
#    # Loop through each run
#    for run_id in range(1, num_runs + 1):
#        file_path = os.path.join(base_sim_dir, f"run_{run_id}", "source_catalog_cutzl04.h5")
#        ax = axes[run_id - 1] # Select the subplot
        
#        if not os.path.exists(file_path):
#            print(f"Warning: File not found for Run {run_id}. Skipping.")
#            ax.set_title(f'Run {run_id} (No Data)')
#            continue
            
#        try:
#            with h5py.File(file_path, "r") as hf:
#                ra_values = hf["RA"][:]
#                dec_values = hf["DEC"][:]
                
#            ax.scatter(ra_values, dec_values, s=0.1, alpha=0.5, c='cyan')
#            ax.set_title(f'Run {run_id} ({len(ra_values)} galaxies)')
#            ax.grid(True, linestyle='--', alpha=0.3)
            
#        except Exception as e:
#            print(f"Error reading file for Run {run_id}: {e}")

#    # Add shared labels and a main title
#    fig.suptitle('Background Galaxy Distribution for Each Run', fontsize=18)
#    fig.text(0.5, 0.04, 'Right Ascension', ha='center', va='center', fontsize=12)
#    fig.text(0.06, 0.5, 'Declination ', ha='center', va='center', rotation='vertical', fontsize=12)
#    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    
#    # Save the plot
#    plot_file_path = os.path.join(output_dir, "background_distributions.png")
#    plt.savefig(plot_file_path, dpi=200)
#    print(f"\nPlot saved to: {plot_file_path}")

#if __name__ == "__main__":
#    plt.style.use('dark_background')
#    plot_separate_backgrounds()



#if __name__ == "__main__":
#    # BG file
#    base_sim_dir = "lhc_run_sims"
#    run_id = 1
#    BG_data = os.path.join(base_sim_dir, f"run_{run_id}", "source_catalog_cutzl04.h5")

#    background_type = "sim"   # or "DES"
#    output_plot = "example_zl04_mesh5e5/background_only.png"

#    # For sim background, need number of rows
#    rows = None
#    if background_type == "sim":
#        with h5py.File(BG_data, "r") as f:
#            rows = f["RA"].shape[0]

#    # Load background
#    bg_ra, bg_dec, g1, g2, z_true, weights = load_background(
#        BG_data, rows=rows, background_type=background_type
#    )

## --- Load filaments ---
#filament_dir = "example_zl04_mesh5e5/filaments"
#filament_h5 = os.path.join(filament_dir, "filaments_p15.h5")    

#with h5py.File(filament_h5, "r") as hdf:
#    dataset = hdf["data"]
#    ra_values = dataset["RA"][:] 
#    dec_values = dataset["DEC"][:]
#    labels = dataset["Filament_Label"][:]

##for label in unique_labels:
##        filament_mask = labels == label
##        filament_coords = np.column_stack((ra_values[filament_mask], dec_values[filament_mask]))
        
        

## === PLOT BACKGROUND + FILAMENTS ===
#plt.figure(figsize=(8, 6))
#plt.scatter(np.radians(bg_ra), np.radians(bg_dec), s=2, c="gray", alpha=0.5)
#plt.scatter(ra_values, dec_values, s=5, c="red", alpha=0.7)
#plt.xlabel("RA ")
#plt.ylabel("DEC")
#plt.title(" Filament Positions")
#plot_file = "example_zl04_mesh5e5/BG-filaments.png"
#plt.savefig(plot_file, dpi=200)
#plt.close()
#print(f"Saved background+filament plot: {plot_file}")
# ===================================

#    # === PLOT BACKGROUND ONLY ===
#    plt.figure(figsize=(8, 6))
#    plt.scatter(bg_ra, bg_dec, s=2, c="blue", alpha=0.5)
#    plt.xlabel("RA [deg]")
#    plt.ylabel("DEC [deg]")
#    plt.title("Background Galaxies (raw from catalog)")
#    plt.savefig(output_plot, dpi=200)
#    plt.close()
#    print(f"Saved background-only plot: {output_plot}")




## --- Diagnostics ---
#print("Filaments RA range:", ra_fil.min(), ra_fil.max())
#print("Filaments DEC range:", dec_fil.min(), dec_fil.max())
#print("Background RA range:", bg_ra.min(), bg_ra.max())
#print("Background DEC range:", bg_dec.min(), bg_dec.max())

## --- Overlay zoomed-in view around the filament patch ---
#plt.figure(figsize=(8,6))
#plt.scatter(bg_ra, bg_dec, s=0.5, c="gray", alpha=0.5, label="Background galaxies")
#plt.scatter(ra_fil, dec_fil, s=5, c="red", label="Filaments")

## Zoom into filament region (slightly padded box around filaments)
#plt.xlim(ra_fil.min()-1, ra_fil.max()+1)
#plt.ylim(dec_fil.min()-1, dec_fil.max()+1)

#plt.xlabel("RA [deg]")
#plt.ylabel("DEC [deg]")
#plt.title("Zoomed-in Overlay: Filaments + Background Galaxies")
#plt.legend()
#plt.show()

#filament_coords = np.column_stack((ra_values, dec_values))
#bg_coords = np.radians(np.column_stack((bg_ra, bg_dec)))


















## --- Background loader functions ---
#def read_sim_background(bg_file, rows):
#    with h5py.File(bg_file, "r") as f:
#        bg_ra = f["RA"][:rows]
#        bg_ra = (bg_ra + 180) % 360
#        bg_dec = f["DEC"][:rows]
#        g1 = f["G1"][:rows]
#        g2 = f["G2"][:rows]
#        z_true = f["Z_TRUE"][:rows]
#        weights = f["weight"][:rows] if "weight" in f else np.ones_like(bg_ra)
#    return bg_ra, bg_dec, g1, g2, z_true, weights

#def read_DES_background(bg_file):
#    with h5py.File(bg_file, "r") as f:
#        bg_ra = f["background"]["ra"][:]
#        bg_ra = (bg_ra + 180) % 360
#        bg_dec = f["background"]["dec"][:]
#        g1 = f["background"]["g1"][:]
#        g2 = f["background"]["g2"][:]
#        weights = f["background"]["weight"][:]
#    return bg_ra, bg_dec, g1, g2, None, weights

#def load_background(bg_file, background_type="sim", rows=None):
#    if background_type == "sim":
#        return read_sim_background(bg_file, rows)
#    elif background_type == "DES":
#        return read_DES_background(bg_file)
#    else:
#        raise ValueError(f"Unknown background_type: {background_type}")


#if __name__ == "__main__":
#    # --- Background catalog ---
#    output_plot_path = "example_zl04_mesh5e5/useful_plots"
#    os.makedirs(output_plot_path, exist_ok=True)
#    base_sim_dir = "lhc_run_sims"
#    run_id = 1
#    BG_data = os.path.join(base_sim_dir, f"run_{run_id}", "source_catalog_cutzl04.h5")
#    background_type = "sim"   # or "DES"
#    rows = None
#    if background_type == "sim":
#        with h5py.File(BG_data, "r") as f:
#            rows = f["RA"].shape[0]

#    bg_ra, bg_dec, g1, g2, z_true, weights = load_background(BG_data, background_type=background_type, rows=rows)
    
#    # --- Plot background only (subsample if large) ---
#    subsample = 50000
#    idx = np.random.choice(len(bg_ra), min(subsample, len(bg_ra)), replace=False)
#    plt.figure(figsize=(8, 6))
#    plt.scatter(bg_ra[idx], bg_dec[idx], s=1, c="blue", alpha=0.5)
#    plt.xlabel("RA [deg]")
#    plt.ylabel("DEC [deg]")
#    plt.title("Background Galaxies (raw)")
#    output_filename = f"background_only.png"
#    output_plot = os.path.join(output_plot_path, output_filename)
#    plt.savefig(output_plot, dpi=200)
#    plt.close()
#    print(f"Saved background-only plot")

#    # --- Filament catalog ---
#    filament_dir = "example_zl04_mesh5e5/filaments"
#    filament_h5 = os.path.join(filament_dir, "filaments_p15.h5")
#    with h5py.File(filament_h5, "r") as hdf:
#        dataset = hdf["data"]
#        ra_values = (dataset["RA"][:] + 180) % 360
#        dec_values = dataset["DEC"][:]
#        labels = dataset["Filament_Label"][:]
    
#    # --- Compute nearest-neighbor distances ---
#    #filament_coords = np.radians(np.column_stack((ra_values, dec_values)))
#    filament_coords = np.column_stack((ra_values, dec_values))
#    bg_coords = np.radians(np.column_stack((bg_ra, bg_dec)))
    
#    nbrs = NearestNeighbors(n_neighbors=1, metric="haversine").fit(filament_coords)
#    distances, indices = nbrs.kneighbors(bg_coords)
    
#    # --- TEMPORARY CHECK ---
#    print(f"Minimum distance: {np.degrees(np.min(distances)) * 60:.2f} arcmin")
#    print(f"Maximum distance: {np.degrees(np.max(distances)) * 60:.2f} arcmin")
    
#    valid_distances = distances[distances > 0]
#    if len(valid_distances) > 0:
#        max_bin_limit = np.percentile(valid_distances, 95)
#        print(f"95th percentile distance: {np.degrees(max_bin_limit) * 60:.2f} arcmin")
    
#    # --- Histogram of distances ---
#    plt.figure(figsize=(8, 6))
#    hist_subsample = np.random.choice(len(valid_distances), min(100000, len(valid_distances)), replace=False)
#    plt.hist(np.degrees(valid_distances[hist_subsample]) * 60, bins=50, color="green", alpha=0.7)
#    plt.xlabel("Distance to nearest filament [arcmin]")
#    plt.ylabel("Number of background galaxies")
#    plt.title("Histogram of Background-to-Filament Distances")
#    hist_plot_filename = f"background_filament_distance_hist.png"
#    hist_plot_path = os.path.join(output_plot_path,hist_plot_filename)
#    plt.savefig(hist_plot_path, dpi=200)
#    plt.close()
#    print(f"Saved histogram plot")
    
#    # --- Combined plot: background + filaments ---
#    subsample_bg = np.random.choice(len(bg_ra), min(subsample, len(bg_ra)), replace=False)
#    bg_plot_ra = bg_ra[subsample_bg]
#    bg_plot_dec = bg_dec[subsample_bg]
#    bg_plot_dist = distances[subsample_bg, 0]  # distance to nearest filament
    
#    plt.figure(figsize=(10, 7))
#    sc = plt.scatter(bg_plot_ra, bg_plot_dec, s=1, c=np.degrees(bg_plot_dist)*60, cmap="viridis", alpha=0.5)
#    plt.colorbar(sc, label="Distance to nearest filament [arcmin]")
#    plt.scatter(ra_values, dec_values, s=5, c="red", alpha=0.7, label="Filaments")
#    plt.xlabel("RA [deg]")
#    plt.ylabel("DEC [deg]")
#    plt.title("Background Galaxies and Filaments (distance-coded)")
#    plt.legend()
#    combined_plot_filename = f"background_filaments_distance.png"
#    combined_plot_path = os.path.join(output_plot_path, combined_plot_filename )
#    plt.savefig(combined_plot_path, dpi=200)
#    plt.close()
#    print(f"Saved combined background+filament plot")





#def read_sim_background(bg_file, stride=1000):
#    """
#    Read background galaxies from simulated catalog (HDF5).
#    Loads the full dataset but only keeps every `stride`-th row.
#    """
#    with h5py.File(bg_file, "r") as f:
#        bg_ra = f["RA"][::stride]
#        bg_ra = (bg_ra + 180) % 360  
#        bg_dec = f["DEC"][::stride]
#        g1 = f["G1"][::stride]
#        g2 = f["G2"][::stride]
#        z_true = f["Z_TRUE"][::stride]
#        weights = f["weight"][::stride] if "weight" in f else np.ones_like(bg_ra)

#    return bg_ra, bg_dec, g1, g2, z_true, weights


## --- Parameters ---
#output_dir = "example_zl04_mesh5e5/usefule_plots"
#data_dir   = "example_zl04_mesh5e5/noise"
#os.makedirs(output_dir, exist_ok=True)

#realization_idx = 0  
#file_path = os.path.join(data_dir, f"source_catalog_noise_{realization_idx:02d}.h5")

## --- Load one realization ---
#ra, dec, g1, g2, z_true, weights = read_sim_background(file_path, stride=500)

## --- Plot galaxies with shear as vectors ---
#plt.figure(figsize=(8, 6))
#plt.quiver(np.radians(ra), np.radians(dec), g1, g2,
#           angles="xy", scale=50, width=0.003, alpha=0.6)
#plt.xlabel("RA")
#plt.ylabel("DEC")
#plt.title(f"Noise Realization {realization_idx}")


## --- Save plot ---
#plot_path = os.path.join(output_dir, f"noise_realization_{realization_idx:02d}.png")
#plt.savefig(plot_path, dpi=200, bbox_inches="tight")
#plt.close()

#print(f"Saved plot: {plot_path}")




##############################################
################ RIDGES PLOTS ##################
##############################################


## --- Configuration ---
#base_label = "zero_err"
#bandwidth = 0.3
#run_id = 1
#fp = 15  # percentile tag 

## ridge file
#home_dir = f"simulation_ridges/{base_label}/band_{bandwidth:.1f}"
#ridges_file = os.path.join(
#    home_dir,
#    f"Ridges_final_p{fp:02d}",
#    f"{base_label}_run_{run_id}_ridges_p{fp:02d}.h5"
#)

#print(f"Loading ridge coordinates from:\n  {ridges_file}")

## --- Load ridges ---
#with h5py.File(ridges_file, "r") as f:
#    ridges = f["ridges"][:]

#print(f"Loaded ridges: {ridges.shape}")

#ra = ridges[:, 1]  
#dec = ridges[:, 0]

## --- scatter plot ---
#plt.figure(figsize=(8, 6))
#plt.scatter(ra, dec, s=0.3, color="red", alpha=0.6)
#plt.xlabel("Right Ascension [rad]")
#plt.ylabel("Declination [rad]")
#plt.title(f"Ridge coordinates – zero_err, run_{run_id}, bw={bandwidth}")
#plt.grid(alpha=0.3)

## --- Save the plot ---
#plot_path = f"plots/ridges_zero_err_run{run_id}_bw{bandwidth}.png"
#plt.savefig(plot_path, bbox_inches="tight", dpi=300)
#plt.close()

#print("Ridge coordinate plot saved to {plot_path}")



##############################################
################ SHEAR PLOTS ##################
##############################################



#import pandas as pd

## === Configuration ===
#filament_dir = "simulation_ridges_comparative_analysis/zero_err/band_0.1/shear_test_run_1"
#fp = 15  # percentile

## === Input files ===
#shear_csv = os.path.join(filament_dir, f"shear_p{fp:02d}.csv")
#shear_flip_csv = os.path.join(filament_dir, f"shear_p{fp:02d}_flipG1.csv")

## === Load the data ===
#def load_shear_data(path):
#    data = pd.read_csv(path)
#    return (
#        data["Bin_Center"],
#        data["Weighted_g_plus"],
#        data["Weighted_g_cross"],
#    )

#r, g_plus, g_cross = load_shear_data(shear_csv)
#r_flip, g_plus_flip, g_cross_flip = load_shear_data(shear_flip_csv)

## === Plot configuration ===
#plt.figure(figsize=(8, 6))
#plt.title("Shear Profiles (Zero-error Shrinked Ridge)")
#plt.plot(r, g_plus, label=r"$g_+$", lw=2)
#plt.plot(r, g_cross, label=r"$g_\times$", lw=2, ls="--")
#plt.plot(r_flip, g_plus_flip, label=r"$g_+^{(\mathrm{flipG1})}$", lw=2, color="C2")
#plt.plot(r_flip, g_cross_flip, label=r"$g_\times^{(\mathrm{flipG1})}$", lw=2, ls="--", color="C3")

#plt.axhline(0, color="gray", lw=1)
#plt.xlabel("Distance [arcmin]")
#plt.ylabel("Weighted Shear")
#plt.legend()
#plt.grid(True, alpha=0.3)
#plt.tight_layout()

## === Save and show ===
#plt.savefig(os.path.join(filament_dir, f"shear_profiles_p{fp:02d}.png"), dpi=200)
#plt.show()

##simulation_ridges_comparative_analysis/zero_err/band_0.1/shear_test_run_1/shear_profiles_p15.png




############################################
########### G1 and G2 ######################
############################################

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- background reader ---
def read_sim_background(bg_file, stride=1000):
    """
    Read background galaxies from simulated catalog (HDF5).
    Loads the full dataset but only keeps every `stride`-th row.
    """
    with h5py.File(bg_file, "r") as f:
        bg_ra = f["RA"][::stride]
        bg_ra = (bg_ra + 180) % 360  
        bg_dec = f["DEC"][::stride]
        g1 = f["G1"][::stride]
        g2 = f["G2"][::stride]
        z_true = f["Z_TRUE"][::stride]
        weights = f["weight"][::stride] if "weight" in f else np.ones_like(bg_ra)

    return bg_ra, bg_dec, g1, g2, z_true, weights


# --- Configuration ---
base_sim_dir = "lhc_run_sims_zero_err_10"
run_id = 1
bg_file = os.path.join(base_sim_dir, f"run_{run_id}", "source_catalog_cutzl04.h5")

output_dir = "simulation_ridges_comparative_analysis/zero_err/band_0.1/shear_test_run_1/useful_plots"
os.makedirs(output_dir, exist_ok=True)

# --- Load data ---
bg_ra, bg_dec, g1, g2, z_true, weights = read_sim_background(bg_file, stride=1000)
print(f"Loaded {len(g1)} background galaxies (every 1000th sample).")

# ============================================================
# Plot 1: g1 vs g2 scatter
# ============================================================
plt.figure(figsize=(6, 6))
plt.scatter(g1, g2, s=3, alpha=0.5, c="royalblue")
plt.axhline(0, color='gray', lw=1, ls='--')
plt.axvline(0, color='gray', lw=1, ls='--')
plt.xlabel(r"$g_1$")
plt.ylabel(r"$g_2$")
plt.title("Shear components $g_1$ vs $g_2$")
plt.grid(True, ls="--", alpha=0.3)
plt.tight_layout()
save_path1 = os.path.join(output_dir, "scatter_g1_vs_g2.png")
plt.savefig(save_path1, dpi=300)
plt.close()
print(f"Saved {save_path1}")

# ============================================================
# Plot 2: RA–Dec map for g1 and g2 (side-by-side subplots)
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

sc1 = ax[0].scatter(bg_ra, bg_dec, c=g1, s=3, cmap="coolwarm", alpha=0.6)
ax[0].set_title(r"$g_1$ distribution")
ax[0].set_xlabel("RA [deg]")
ax[0].set_ylabel("Dec [deg]")
plt.colorbar(sc1, ax=ax[0], label=r"$g_1$")

sc2 = ax[1].scatter(bg_ra, bg_dec, c=g2, s=3, cmap="coolwarm", alpha=0.6)
ax[1].set_title(r"$g_2$ distribution")
ax[1].set_xlabel("RA [deg]")
plt.colorbar(sc2, ax=ax[1], label=r"$g_2$")

plt.tight_layout()
save_path2 = os.path.join(output_dir, "sky_g1_g2_maps.png")
plt.savefig(save_path2, dpi=300)
plt.close()
print(f"Saved {save_path2}")


