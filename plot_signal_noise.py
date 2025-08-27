import os
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
filament_dir = "example_zl04_mesh5e5/filaments"
noise_shear_dir = "example_zl04_mesh5e5/noise/shear"
plot_dir =  "example_zl04_mesh5e5/shear_plots"
final_percentile = 15
num_realizations = 30

# --- Input files ---
shear_csv = os.path.join(filament_dir, f"shear_p{final_percentile:02d}.csv")
noise_files = [
    os.path.join(noise_shear_dir, f"shear_noise_p{final_percentile:02d}_r{i:02d}.csv")
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
plt.savefig(os.path.join(plot_dir, "noise_only.png"), dpi=200)
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
plt.savefig(os.path.join(plot_dir, "signal_vs_noise.png"), dpi=200)
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
plt.savefig(os.path.join(plot_dir, "signal_minus_noise.png"), dpi=200)
plt.show()

print("Plots saved: noise_only.png, signal_vs_noise.png, signal_minus_noise.png")

