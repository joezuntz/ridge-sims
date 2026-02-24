import os
import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# -------------------------------------------------
# PATHS
# -------------------------------------------------
#current_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))

#ridge_file = os.path.join(
#    current_dir,
#    "DES_fiducial_sim/band_0.1_mesh_2/Ridges_final_p15/shape_err_run_1_ridges_p15_contracted.h5"
#)

#bg_file = os.path.join(
#    parent_dir,
#    "lhc_DES_fiducial_sim",
#    "run_1",
#    "source_catalog_cutzl0.40.h5"
#)

#out_png = "sim_background_with_ridges.png"

# -------------------------------------------------
# LOAD BACKGROUND 
# -------------------------------------------------
#with h5py.File(bg_file, "r") as f:
#    ra_bg  = f["RA"][:]        # degrees
#    dec_bg = f["DEC"][:]       # degrees

#ra_bg = (ra_bg + 180.0) % 360.0   # REQUIRED SHIFT

# -------------------------------------------------
# LOAD RIDGES (radians → degrees)
# -------------------------------------------------
#with h5py.File(ridge_file, "r") as f:
#    ridges = f["ridges"][:]

#dec_r = np.degrees(ridges[:, 0])
#ra_r  = np.degrees(ridges[:, 1])

# -------------------------------------------------
#  NUMERICAL CHECK 
# -------------------------------------------------
#print("[CHECK] Background RA range :", ra_bg.min(), ra_bg.max())
#print("[CHECK] Background DEC range:", dec_bg.min(), dec_bg.max())
#print("[CHECK] Ridges RA range     :", ra_r.min(), ra_r.max())
#print("[CHECK] Ridges DEC range    :", dec_r.min(), dec_r.max())

# -------------------------------------------------
# PLOT
# -------------------------------------------------
#plt.figure(figsize=(10, 6.5))

#H, xedges, yedges = np.histogram2d(
#    ra_bg, dec_bg,
#    bins=[1200, 800]
#)
#H = np.log10(H + 1)

#plt.imshow(
#    H.T,
#    origin="lower",
#    aspect="auto",
#    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
#    cmap="Greys"
#)

#plt.scatter(
#    ra_r, dec_r,
#    s=0.2,
#    c="red",
#    alpha=0.9,
#    label="Contracted ridges"
#)

#plt.xlabel("RA [deg]")
#plt.ylabel("DEC [deg]")
#plt.title("Simulation background + contracted ridges (shifted)")
#plt.legend(markerscale=10)
#plt.tight_layout()
#plt.savefig(out_png, dpi=250)
#plt.close()

#print(f"[OK] Saved → {out_png}")



##########################################################

# --------------- Shear test plots -----------------------

##########################################################




current_dir = os.path.dirname(os.path.abspath(__file__))

csv_file2 = os.path.join(current_dir, "density_threshold_test/shear_vs_threshold/run_1_shear_fp_40.csv")
#csv_file2 = os.path.join(current_dir, "shear/DES_fiducial_sim/band_0.1_mesh_5/run_1_p15/signal_shear.csv")
#csv_file  = os.path.join(current_dir, "shear_no_endpoint/signal_shear.csv")
out_png   = os.path.join(current_dir, "density_threshold_test", "gamma_plus_fp_40.png")

#if not os.path.exists(csv_file):
#    raise FileNotFoundError(csv_file)

# load (skip header row)
d2 = np.loadtxt(csv_file2, delimiter=",", skiprows=1)
#d  = np.loadtxt(csv_file,  delimiter=",", skiprows=1)

# --- no-endpoint ---
#real_dist  = d[:, 1]
#g_plus     = d[:, 2]
#x_arcmin   = np.degrees(real_dist) * 60.0

# --- full filaments ---
real_dist2 = d2[:, 1]
g_plus2    = d2[:, 2]
x_arcmin2  = np.degrees(real_dist2) * 60.0

plt.figure(figsize=(7.5, 6.0))
#plt.plot(x_arcmin,  g_plus,  marker="o", lw=1.5, ms=4, label="No endpoints")
plt.plot(x_arcmin2, g_plus2, marker="s", lw=1.5, ms=4)#, label="All filaments")

plt.xscale("log")
plt.xlabel("Angular separation [arcmin]")
plt.ylabel(r"$\gamma_{+}$")
plt.title(r"shear profile $\gamma_{+}$")
#plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig(out_png, dpi=250)
plt.close()

print(f"[OK] Saved → {out_png}")



############# ------------- X^2 Computation --------------------- ####################


#sim_shear_dir = "shear/DES_fiducial_sim/band_0.1_mesh_2/run_1_p15"
#sim_shear_csv = os.path.join(sim_shear_dir, "signal_shear.csv")


#sim_noise_files = sorted(glob.glob(os.path.join(sim_shear_dir, "random_rotations/shear_random_*.csv")))

## Cov regularization 
#def inv_cov(cov, eps=1e-12):
#    try:
#        return np.linalg.inv(cov)
#    except np.linalg.LinAlgError:
#        print("[WARN] Covariance singular -> adding diagonal regularization.")
#        cov_reg = cov + np.eye(cov.shape[0]) * eps
#        return np.linalg.inv(cov_reg)




#def run_analysis(case_label, shear_csv, noise_files):
#    print(f"\n=== Running analysis for {case_label} ===")


#    # --- Load signal ---
#    signal_data = np.loadtxt(shear_csv, delimiter=",", skiprows=1)
#    bin_center_rad = signal_data[:, 0]
#    arcmin_centers = np.degrees(bin_center_rad) * 60.0

#    g_plus_signal = signal_data[:, 2]
#    g_cross_signal = signal_data[:, 3]

#    # --- Load noise realizations ---
#    all_g_plus_noise = []
#    all_g_cross_noise = []

#    used_noise = 0
#    for nf in noise_files:
#        if not os.path.exists(nf):
#            continue
#        data = np.loadtxt(nf, delimiter=",", skiprows=1)
#        all_g_plus_noise.append(data[:, 2])
#        all_g_cross_noise.append(data[:, 3])
#        used_noise += 1


#    all_g_plus_noise = np.array(all_g_plus_noise)
#    all_g_cross_noise = np.array(all_g_cross_noise)


#    # --- Noise mean/std ---
#    g_plus_noise_mean = np.mean(all_g_plus_noise, axis=0)
#    g_cross_noise_mean = np.mean(all_g_cross_noise, axis=0)

#    g_plus_noise_std = np.std(all_g_plus_noise, axis=0, ddof=1)
#    g_cross_noise_std = np.std(all_g_cross_noise, axis=0, ddof=1)


#    # --- Covariance matrices from noise realizations ---
#    cov_plus = np.cov(all_g_plus_noise, rowvar=False, ddof=1)
#    cov_cross = np.cov(all_g_cross_noise, rowvar=False, ddof=1)

#    cov_plus_inv = inv_cov(cov_plus)
#    cov_cross_inv = inv_cov(cov_cross)

#    # --- Hartlap factor ---
#    N = used_noise
#    p = cov_plus.shape[0]
    
#    if N > p + 2:
#        hartlap = (N - p - 2) / (N - 1)
#        cov_plus_inv  = hartlap * cov_plus_inv
#        cov_cross_inv = hartlap * cov_cross_inv
#    else:
#        print(f"Hartlap invalid (N={N}, p={p}). Skipping Hartlap correction.")
    
#    # --- Chi-square ---
#    d_plus  = g_plus_signal
#    d_cross = g_cross_signal
#    dof = len(d_plus)
    
#    chi2_plus  = float(d_plus  @ cov_plus_inv  @ d_plus)
#    chi2_cross = float(d_cross @ cov_cross_inv @ d_cross)
    
#    print(f"[g_plus]  chi2 = {chi2_plus:.3f}, dof = {dof}, chi2/dof = {chi2_plus/dof:.3f}")
#    print(f"[g_cross] chi2 = {chi2_cross:.3f}, dof = {dof}, chi2/dof = {chi2_cross/dof:.3f}")
    
#    # p-values without
#    pval_plus  = stats.chi2.sf(chi2_plus,  dof)
#    pval_cross = stats.chi2.sf(chi2_cross, dof)
    
#    print(f"[g_plus]  p = {pval_plus:.3e}")
#    print(f"[g_cross] p = {pval_cross:.3e}")
    
    
    
#if __name__ == "__main__":


#    run_analysis(
#        case_label="DES fiducial sim (run_1_p15)",
#        shear_csv=sim_shear_csv,
#        noise_files=sim_noise_files
#    )
    
