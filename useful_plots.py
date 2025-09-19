import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random

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
    


# --- config ---
base_dir = "example30_band0.4/8test"
noise_dir = os.path.join(base_dir, "noise_data")
output_dir = os.path.join(base_dir, "useful_plots")
os.makedirs(output_dir, exist_ok=True)

# how many random files to inspect per run
num_samples = 1
runs_to_check = [1, 2, 3]

for run_id in runs_to_check:
    for _ in range(num_samples):
        noise_idx = random.randint(0, 299)
        noise_file = os.path.join(noise_dir, f"noise_r{run_id:02d}_n{noise_idx:03d}.h5")

        if not os.path.exists(noise_file):
            print(f"[Run {run_id}] File not found: {noise_file}")
            continue

        print(f"\n[Run {run_id}] Inspecting {noise_file}")
        with h5py.File(noise_file, "r") as f:
            keys = list(f.keys())
            print("  Keys in file:", keys)

            try:
                dec = f["DEC"][:]
                ra = f["RA"][:]
                g1 = f["G1"][:]
                g2 = f["G2"][:]
            except KeyError as e:
                print(f"  Missing expected dataset: {e}")
                continue

        print(f"  -> Catalog size: {len(ra)} objects")

        # scatter plot RA/Dec
        plt.figure(figsize=(6, 4))
        plt.scatter(ra, dec, s=0.2, alpha=0.5)
        plt.xlabel("RA")
        plt.ylabel("Dec")
        plt.title(f"Run {run_id} – Noise n{noise_idx:03d} (RA/Dec)")
        plt.tight_layout()
        out_png = os.path.join(output_dir, f"run{run_id}_noise{noise_idx:03d}_radec.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"  -> RA/Dec plot saved to {out_png}")

        # histogram of g1/g2
        plt.figure(figsize=(6, 4))
        plt.hist(g1, bins=100, alpha=0.6, label="G1")
        plt.hist(g2, bins=100, alpha=0.6, label="G2")
        plt.xlabel("Shear value")
        plt.ylabel("Count")
        plt.title(f"Run {run_id} – Noise n{noise_idx:03d} (G1/G2)")
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(output_dir, f"run{run_id}_noise{noise_idx:03d}_g1g2.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"  -> G1/G2 histogram saved to {out_png}")