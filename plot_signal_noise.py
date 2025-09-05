import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- Configuration ---
#filament_dir = "example_zl04_mesh5e5/filaments"
#noise_shear_dir = "example_zl04_mesh5e5/noise/shear"
#plot_dir =  "example_zl04_mesh5e5/shear_plots"

filament_dir = "example_zl04_mesh5e5/filaments/shrinked_filaments" 
noise_shear_dir = "example_zl04_mesh5e5/noise/shrinked_ridges_shear"
plot_dir =  "example_zl04_mesh5e5/shrinked_shear_plots"
os.makedirs(plot_dir, exist_ok=True)
final_percentile = 15
num_realizations = 300

# --- Input files ---
shear_csv = os.path.join(filament_dir, f"shear_p{final_percentile:02d}.csv")
if not os.path.exists(shear_csv):
    shear_csv = os.path.join(filament_dir, f"shear_p{final_percentile:02d}_flipG1.csv")

# derive suffix for plots based on filename
plot_suffix = "_flipG1" if "flipG1" in shear_csv else ""

noise_files = [
    os.path.join(noise_shear_dir, f"shear_noise_p{final_percentile:02d}_r{i:02d}{plot_suffix}.csv")
    for i in range(num_realizations)
]

# --- Load real signal ---
signal_data = np.loadtxt(shear_csv, delimiter=",", skiprows=1)
bin_center_rad = signal_data[:, 0]
arcmin_centers = np.degrees(bin_center_rad) * 60.0
g_plus_signal = signal_data[:, 2]
g_cross_signal = signal_data[:, 3]

# --- Load noise realizations ---
all_g_plus_noise = []
all_g_cross_noise = []

for nf in noise_files:
    if not os.path.exists(nf):
        print(f"[Warning] Missing noise file: {nf}")
        continue
    data = np.loadtxt(nf, delimiter=",", skiprows=1)
    all_g_plus_noise.append(data[:, 2])
    all_g_cross_noise.append(data[:, 3])

all_g_plus_noise = np.array(all_g_plus_noise)  
all_g_cross_noise = np.array(all_g_cross_noise)

# --- Compute noise mean and std ---
g_plus_noise_mean = np.mean(all_g_plus_noise, axis=0)
g_cross_noise_mean = np.mean(all_g_cross_noise, axis=0)
g_plus_noise_std = np.std(all_g_plus_noise, axis=0)
g_cross_noise_std = np.std(all_g_cross_noise, axis=0)

# --- Subtract noise ---
g_plus_subtracted = g_plus_signal - g_plus_noise_mean
g_cross_subtracted = g_cross_signal - g_cross_noise_mean


# --- Covariance matrices from noise realizations ---
cov_plus = np.cov(all_g_plus_noise, rowvar=False)   # shape (bins, bins)
cov_cross = np.cov(all_g_cross_noise, rowvar=False)

# --- Data vectors (signal - noise mean) ---
d_plus = g_plus_subtracted
d_cross = g_cross_subtracted

# --- Model (zero signal) ---
m_plus = np.zeros_like(d_plus)
m_cross = np.zeros_like(d_cross)

# --- Inverse covariances ---
cov_plus_inv = np.linalg.inv(cov_plus)
cov_cross_inv = np.linalg.inv(cov_cross)

# --- Chi-square ---
chi2_plus = (d_plus - m_plus) @ cov_plus_inv @ (d_plus - m_plus)
chi2_cross = (d_cross - m_cross) @ cov_cross_inv @ (d_cross - m_cross)

# Degrees of freedom = number of bins
dof = len(d_plus)

# --- p-values ---
pval_plus = 1 - stats.chi2.cdf(chi2_plus, dof)
pval_cross = 1 - stats.chi2.cdf(chi2_cross, dof)

# --- Convert p-values to sigma (Gaussian equivalent) ---
sigma_plus = stats.norm.isf(pval_plus/2)   # two-tailed to sigma
sigma_cross = stats.norm.isf(pval_cross/2)

print(f"[g_plus]  chi2 = {chi2_plus:.2f}  dof = {dof}  p = {pval_plus:.2e}  ~ {sigma_plus:.2f}σ detection")
print(f"[g_cross] chi2 = {chi2_cross:.2f}  dof = {dof}  p = {pval_cross:.2e}  ~ {sigma_cross:.2f}σ detection")


# --- Plot noise only (with error bars) ---
plt.figure(figsize=(7,5))
plt.errorbar(arcmin_centers, g_plus_noise_mean, yerr=g_plus_noise_std, fmt='o-', label=r"$g_+$ noise")
plt.errorbar(arcmin_centers, g_cross_noise_mean, yerr=g_cross_noise_std, fmt='x-', label=r"$g_\times$ noise")
plt.xscale("log")
plt.xlabel("Separation (arcmin)")
plt.ylabel("Shear (noise)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"noise_only{plot_suffix}.png"), dpi=200)
plt.show()

# --- Plot signal vs noise-subtracted ---
fig, ax = plt.subplots(1, 2, figsize=(14,5), sharex=True)

# g_plus
ax[0].plot(arcmin_centers, g_plus_signal, 'k--', label="Raw signal $g_+$")
ax[0].errorbar(arcmin_centers, g_plus_subtracted, yerr=g_plus_noise_std, fmt='o-', label="Signal - Noise")
ax[0].set_xscale("log")
ax[0].set_xlabel("Separation (arcmin)")
ax[0].set_ylabel(r"$g_+$")
ax[0].grid(True, which="both", ls="--")
ax[0].legend()

# g_cross
ax[1].plot(arcmin_centers, g_cross_signal, 'k--', label="Raw signal $g_\times$")
ax[1].errorbar(arcmin_centers, g_cross_subtracted, yerr=g_cross_noise_std, fmt='x-', label="Signal - Noise")
ax[1].set_xscale("log")
ax[1].set_xlabel("Separation (arcmin)")
ax[1].set_ylabel(r"$g_\times$")
ax[1].grid(True, which="both", ls="--")
ax[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"signal_vs_noise{plot_suffix}.png"), dpi=200)
plt.show()

print("Plots saved: noise_only.png and signal_vs_noise.png")


# --- Plot noise-subtracted only (with error bars) ---
plt.figure(figsize=(7,5))
plt.errorbar(
    arcmin_centers, g_plus_subtracted, yerr=g_plus_noise_std,
    fmt='o-', label=r"$g_+$ (signal - noise)"
)
plt.errorbar(
    arcmin_centers, g_cross_subtracted, yerr=g_cross_noise_std,
    fmt='x-', label=r"$g_\times$ (signal - noise)"
)
plt.xscale("log")
plt.xlabel("Separation (arcmin)")
plt.ylabel("Shear (signal - noise)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"signal_minus_noise{plot_suffix}.png"), dpi=200)
plt.show()

print("Plots saved: noise_only.png, signal_vs_noise.png, signal_minus_noise.png")


# --- Plot noise vs noise-subtracted signal together ---
plt.figure(figsize=(7,5))

# Noise with error bars
plt.errorbar(
    arcmin_centers, g_plus_noise_mean, yerr=g_plus_noise_std,
    fmt='o--', color="gray", label=r"$g_+$ noise"
)
plt.errorbar(
    arcmin_centers, g_cross_noise_mean, yerr=g_cross_noise_std,
    fmt='x--', color="lightgray", label=r"$g_\times$ noise"
)

# Signal - Noise with error bars
plt.errorbar(
    arcmin_centers, g_plus_subtracted, yerr=g_plus_noise_std,
    fmt='o-', color="blue", label=r"$g_+$ (signal - noise)"
)
plt.errorbar(
    arcmin_centers, g_cross_subtracted, yerr=g_cross_noise_std,
    fmt='x-', color="red", label=r"$g_\times$ (signal - noise)"
)

plt.xscale("log")
plt.xlabel("Separation (arcmin)")
plt.ylabel("Shear")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"noise_vs_signal_minus_noise{plot_suffix}.png"), dpi=200)
plt.show()

print("Plots saved: noise_only.png, signal_vs_noise.png, signal_minus_noise.png, noise_vs_signalminusnoise.png")


# --- Plot covariance matrices ---
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

im0 = ax[0].imshow(cov_plus, cmap="viridis", origin="lower")
ax[0].set_title(r"Covariance $g_+$")
ax[0].set_xlabel("Bin")
ax[0].set_ylabel("Bin")
fig.colorbar(im0, ax=ax[0], shrink=0.8)

im1 = ax[1].imshow(cov_cross, cmap="viridis", origin="lower")
ax[1].set_title(r"Covariance $g_\times$")
ax[1].set_xlabel("Bin")
ax[1].set_ylabel("Bin")
fig.colorbar(im1, ax=ax[1], shrink=0.8)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"covariance_matrices{plot_suffix}.png"), dpi=200)
plt.close()

print(f"Saved covariance heatmaps to {plot_dir}/covariance_matrices{plot_suffix}.png")
