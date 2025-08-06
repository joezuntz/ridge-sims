import os
import numpy as np
import matplotlib.pyplot as plt


filament_dir = "example/filaments"
final_percentiles = [0, 10, 25, 40, 50, 60, 75, 85, 90, 95]

# Plot configuration
plt.style.use("seaborn-v0_8-darkgrid")
colors = plt.cm.viridis(np.linspace(0, 1, len(final_percentiles)))

fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
ax[0].set_title("$g_+$ vs Distance")
ax[1].set_title("$g_×$ vs Distance")

for i, fp in enumerate(final_percentiles):
    shear_csv = os.path.join(filament_dir, f"shear_p{fp:02d}.csv")
    if not os.path.exists(shear_csv):
        print(f"[Warning] File not found: {shear_csv}")
        continue

    data = np.loadtxt(shear_csv, delimiter=",", skiprows=1)
    bin_center = data[:, 0]  # midpoint of bin can be changed to weighted_real_distance 
    weighted_g_plus = data[:, 2]
    weighted_g_cross = data[:, 3]
    counts = data[:, 4]

    label = f"{fp}th pct"
    ax[0].plot(bin_center, weighted_g_plus, marker='o', color=colors[i], label=label)
    ax[1].plot(bin_center, weighted_g_cross, marker='x', color=colors[i], label=label)

for a in ax:
    a.set_xlabel("Angular Distance [radians]")
    a.set_ylabel("Shear")
    a.axhline(0, color='black', lw=0.5, ls='--')
    a.legend(fontsize="small")

plt.tight_layout()
plt.savefig(os.path.join(filament_dir, "shear_profiles.png"))
plt.show()
