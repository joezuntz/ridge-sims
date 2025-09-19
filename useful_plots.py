import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- config ---
base_dir = "example30_band0.4/8test"
num_runs = 8
final_percentile = 15

for run_id in range(1, num_runs + 1):
    ridge_file = os.path.join(
        base_dir, f"run_{run_id}", "ridges_filtered", f"ridges_p{final_percentile:02d}_filtered.h5"
    )
    if not os.path.exists(ridge_file):
        print(f"[Run {run_id}] No file found: {ridge_file}")
        continue

    with h5py.File(ridge_file, "r") as f:
        ridges = f["ridges"][:]

    if ridges.size == 0:
        print(f"[Run {run_id}] File exists but contains no ridges")
        continue

    ridge_dec = ridges[:, 0]
    ridge_ra = ridges[:, 1]

    plt.figure(figsize=(6, 4))
    plt.scatter(ridge_ra, ridge_dec, s=1, alpha=0.6)
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.title(f"Run {run_id} – Filtered ridges (p{final_percentile})")
    plt.tight_layout()
    
    output_plot = os.path.join(base_dir, "useful_plots"
    out_png = ridge_file.replace(".h5", ".png")
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"[Run {run_id}] Plot saved → {out_png}")
