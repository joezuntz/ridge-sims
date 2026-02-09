import os
import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats







## ============================================================
## PATHS (run from DESY3/)
## ============================================================
#current_dir = os.path.dirname(os.path.abspath(__file__))

#fil_dir = os.path.join(current_dir, "filaments")
#out_dir = os.path.join(current_dir, "test_plots")
#os.makedirs(out_dir, exist_ok=True)




# ============================================================
# PATHS (run from DESY3/)
# ============================================================
#current_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))

#base_catalog_dir = os.path.join(parent_dir, "des-data")
#bg_file = os.path.join(
#    base_catalog_dir,
#    "des-y3-ridges-background-v2_cutzl0.40.h5"
#)

#out_dir = os.path.join(current_dir, "test_plots")
#os.makedirs(out_dir, exist_ok=True)

## ============================================================
## MAIN
## ============================================================
#def main():

#    if not os.path.exists(bg_file):
#        raise FileNotFoundError(f"Background file not found:\n{bg_file}")

#    with h5py.File(bg_file, "r") as f:
#        ra  = f["ra"][:]      
#        dec = f["dec"][:]     

#        #check
#        g1 = f["g1"][:]
#        g2 = f["g2"][:]
#        z  = f["z"][:]

#    print(f"[INFO] Loaded background:")
#    print(f"       N = {len(ra)}")
#    print(f"       ra range  = [{ra.min()}, {ra.max()}]")
#    print(f"       dec range = [{dec.min()}, {dec.max()}]")
#    print(f"       z range   = [{z.min()}, {z.max()}]")

#    out_png = os.path.join(out_dir, "background_raw.png")

#    plt.figure(figsize=(8, 6))
#    plt.scatter(ra, dec, s=0.2, alpha=0.4)
#    plt.xlabel("ra ")
#    plt.ylabel("dec")
#    plt.title("DES Y3 background")
#    plt.grid(True, which="both", ls="--", alpha=0.3)
#    plt.tight_layout()
#    plt.savefig(out_png, dpi=200)
#    plt.close()

#    print(f"[OK] Saved plot → {out_png}")


#if __name__ == "__main__":
#    main()
    
    
    
#"""
#info from back ground 
#N = 107695056
#ra range  = [118.41659481100214, 280.5889403050496]
#dec range = [-68.15964514701932, 5.83360580072365]
#z range   = [0.40000003576278687, 111.53411102294922]

#"""







## ============================================================
## PATHS
## ============================================================
#current_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))

## DES background (degrees in file)
#bg_file = os.path.join(parent_dir, "des-data", "des-y3-ridges-background-v2_cutzl0.40.h5")

## contracted ridges 
#ridge_file = os.path.join(
#    current_dir,
#    "DES_ridge_analysis",
#    "Ridges_analysis",
#    "DESY3_ridges_p15__mesh2_band0.10_contracted_update.h5"
#)

#out_dir = os.path.join(current_dir, "test_plots")
#os.makedirs(out_dir, exist_ok=True)


#def main():
#    if not os.path.exists(bg_file):
#        raise FileNotFoundError(f"Background file not found:\n{bg_file}")
#    if not os.path.exists(ridge_file):
#        raise FileNotFoundError(f"Contracted ridge file not found:\n{ridge_file}")

#    # ------------------------------------------------------------
#    # Load background RA/DEC (DEG) and apply the SAME shift used in ridge finding
#    # ------------------------------------------------------------
#    with h5py.File(bg_file, "r") as f:
#        ra_bg  = f["ra"][:]
#        dec_bg = f["dec"][:]

#    #ra_bg = (ra_bg + 180.0) % 360.0  # shift to match ridge-finder convention

#    # ------------------------------------------------------------
#    # Load contracted ridges (RAD) -> convert to DEG
#    # ------------------------------------------------------------
#    with h5py.File(ridge_file, "r") as f:
#        ridges = f["ridges"][:]

#    dec_r = np.degrees(ridges[:, 0])  # NEW: radians -> degrees
#    ra_r  = np.degrees(ridges[:, 1])  # NEW: radians -> degrees

#    # ------------------------------------------------------------
#    # Define plotting region from ridges
#    # ------------------------------------------------------------
#    pad_deg = 1.0
#    ra_min, ra_max = ra_r.min() - pad_deg, ra_r.max() + pad_deg
#    dec_min, dec_max = dec_r.min() - pad_deg, dec_r.max() + pad_deg

#    # if span is huge, just use full range
#    if (ra_max - ra_min) > 300:
#        ra_min, ra_max = 0.0, 360.0

#    # ------------------------------------------------------------
#    # Background density image in the same window
#    # ------------------------------------------------------------
#    bins_ra = 800
#    bins_dec = 500

#    H, xedges, yedges = np.histogram2d(
#        ra_bg, dec_bg,
#        bins=[bins_ra, bins_dec],
#        range=[[ra_min, ra_max], [dec_min, dec_max]],
#    )
#    H = np.log10(H + 1.0)

#    # ------------------------------------------------------------
#    # Plot
#    # ------------------------------------------------------------
#    out_png = os.path.join(out_dir, "background_density_with_contracted_ridges_update_noshift.png")

#    plt.figure(figsize=(10, 6.5))
#    plt.imshow(
#        H.T,
#        origin="lower",
#        aspect="auto",
#        extent=[ra_min, ra_max, dec_min, dec_max],
#    )

#    plt.scatter(ra_r, dec_r, s=0.2, alpha=0.9)
    
#    plt.scatter(ra_bg, dec_bg, s=0.05, alpha=0.02, color="gray")

#    plt.xlabel("RA [deg] (shifted)")
#    plt.ylabel("DEC [deg]")
#    plt.title("DES background density (shifted) + contracted ridges (update)")
#    plt.tight_layout()
#    plt.savefig(out_png, dpi=250)
#    plt.close()

#    # ------------------------------------------------------------
#    # Numeric overlap sanity check
#    # ------------------------------------------------------------
#    inside = (
#        (ra_r >= ra_bg.min()) & (ra_r <= ra_bg.max()) &
#        (dec_r >= dec_bg.min()) & (dec_r <= dec_bg.max())
#    )
#    frac_inside = inside.mean()

#    print(f"[OK] Saved → {out_png}")
#    print(f"[CHECK] Ridge points inside BG RA/DEC bounding box: {frac_inside*100:.2f}%")
#    print(f"[INFO] BG(shifted) RA range  = [{ra_bg.min():.3f}, {ra_bg.max():.3f}]")
#    print(f"[INFO] BG DEC range          = [{dec_bg.min():.3f}, {dec_bg.max():.3f}]")
#    print(f"[INFO] Ridges RA range       = [{ra_r.min():.3f}, {ra_r.max():.3f}]")
#    print(f"[INFO] Ridges DEC range      = [{dec_r.min():.3f}, {dec_r.max():.3f}]")


#if __name__ == "__main__":
#    main()



#current_dir = os.path.dirname(os.path.abspath(__file__))

#csv_file = os.path.join(current_dir, "shear_flipg2", "shear_p15_flipg2.csv")
#out_png  = os.path.join(current_dir, "shear_flipg2", "gamma_plus_noshift_p15_flipg2_Rcal.png")

#if not os.path.exists(csv_file):
#    raise FileNotFoundError(csv_file)

## load (skip header row)
#d = np.loadtxt(csv_file, delimiter=",", skiprows=1)

#bin_center = d[:, 0]          # radians
#real_dist  = d[:, 1]          # radians 
#g_plus     = d[:, 2]

## plot vs distance in arcmin 
#x_arcmin = np.degrees(real_dist) * 60.0

#plt.figure(figsize=(7.5, 6.0))
#plt.plot(x_arcmin, g_plus, marker="o", lw=1.5, ms=4)
#plt.xscale("log")
#plt.xlabel("Angular separation [arcmin]")
#plt.ylabel(r"$\gamma_{+}$")
#plt.title(r"DES Y3 shear profile (p15): $\gamma_{+}$")
#plt.grid(True, which="both", ls="--", alpha=0.3)
#plt.tight_layout()
#plt.savefig(out_png, dpi=250)
#plt.close()

#print(f"[OK] Saved → {out_png}")







############# ------------- X^2 Computation --------------------- ####################


DESY3_shear_dir = "DESY3/shear_flipg2"
DESY3_shear_csv = os.path.join(DESY3_shear_dir, "shear_p15_flipg2.csv")


DESY3_noise_files = sorted(glob.glob(os.path.join(DESY3_shear_dir, "shear_random_p15_*.csv")))  

# Cov regularization 
def inv_cov(cov, eps=1e-12):
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        print("[WARN] Covariance singular -> adding diagonal regularization.")
        cov_reg = cov + np.eye(cov.shape[0]) * eps
        return np.linalg.inv(cov_reg)




def run_analysis(case_label, shear_csv, noise_files):
    print(f"\n=== Running analysis for {case_label} ===")


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
    N = used_noise
    p = cov_plus.shape[0]
    
    if N > p + 2:
        hartlap = (N - p - 2) / (N - 1)
        cov_plus_inv  = hartlap * cov_plus_inv
        cov_cross_inv = hartlap * cov_cross_inv
    else:
        print(f"Hartlap invalid (N={N}, p={p}). Skipping Hartlap correction.")
    
    # --- Chi-square ---
    d_plus  = g_plus_signal
    d_cross = g_cross_signal
    dof = len(d_plus)
    
    chi2_plus  = float(d_plus  @ cov_plus_inv  @ d_plus)
    chi2_cross = float(d_cross @ cov_cross_inv @ d_cross)
    
    print(f"[g_plus]  chi2 = {chi2_plus:.3f}, dof = {dof}, chi2/dof = {chi2_plus/dof:.3f}")
    print(f"[g_cross] chi2 = {chi2_cross:.3f}, dof = {dof}, chi2/dof = {chi2_cross/dof:.3f}")
    
    # p-values without
    pval_plus  = stats.chi2.sf(chi2_plus,  dof)
    pval_cross = stats.chi2.sf(chi2_cross, dof)
    
    print(f"[g_plus]  p = {pval_plus:.3e}")
    print(f"[g_cross] p = {pval_cross:.3e}")
    
    
    
if __name__ == "__main__":


    run_analysis(
        case_label="DESY3",
        shear_csv=DESY3_shear_csv,
        noise_files=DESY3_noise_files
    )
    

