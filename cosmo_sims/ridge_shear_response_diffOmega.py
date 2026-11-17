
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path("Cosmo_sim2_ridges/sigma8_fixed")


def read_run(number):
    path = root / f"run_{number}/band_0.1/Ridges_final_p15/shear/shear_p15.csv"
    data = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)
    data = np.sort(data, order="Bin_Center")
    measured = (data["Counts"] > 0) & (data["bin_weight"] > 0)
    return data["Bin_Center"], np.where(measured, data["Weighted_g_plus"], np.nan)


bins4, g4 = read_run(4)   # Omega_m = 0.293333
bins6, g6 = read_run(6)   # Omega_m = 0.300000 (fiducial)
bins5, g5 = read_run(5)   # Omega_m = 0.311111


# Omega_m grid as it was made in Cosmo_sims_slurm.py.
grid = np.linspace(0.24, 0.40, 10)
grid[5] = 0.3
omega_m = grid[[3, 5, 4]]                 # increasing Omega_m
g_plus = np.stack((g4, g6, g5))          # rows: cosmologies, columns: angular bins

# Differentiate
dg_domega = np.gradient(g_plus, omega_m, axis=0)
response = dg_domega[1]    # d g_plus / d Omega_m   
theta_arcmin = np.rad2deg(bins6) * 60

output = Path("ridge_response_results/omega_m_mesh1E6_response")
output.parent.mkdir(exist_ok=True)
np.savetxt(output.with_suffix(".csv"), np.column_stack((theta_arcmin, g4, g6, g5, response)),
           delimiter=",", comments="",
           header="theta_arcmin,g_plus_run_4,g_plus_run_6,g_plus_run_5,d_g_plus_d_omega_m")

plt.semilogx(theta_arcmin, response, "o-")
plt.axhline(0, color="grey", linewidth=0.8)
plt.xlabel(r"Separation $\theta$ [arcmin]")
plt.ylabel(r"$\partial \langle g_+\rangle / \partial \Omega_m$")
plt.title(r"Fixed $\sigma_8=0.8$, $10^6$ mesh points")
plt.tight_layout()
plt.savefig(output.with_suffix(".pdf"))
print(f"Saved the CSV and PDF in {output.parent}/")
