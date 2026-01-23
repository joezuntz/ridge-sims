import os
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt

# PATHS

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))

# Background catalog
bg_file = os.path.join(
    parent_dir,
    "lhc_run_sims/run_1",
    "source_catalog_cutzl04.h5"
)

# Filaments from shear pipeline
filament_h5 = os.path.join(
    current_dir,
    "shear",
    "run_1",
    "band_0.1",
    "mesh_2",
    "run_1_p15_filaments.h5"
)

fil_dataset_name = "data"

# Output
out_dir = os.path.join(current_dir, "test_plots")
os.makedirs(out_dir, exist_ok=True)



def main():
    if not os.path.exists(bg_file):
        raise FileNotFoundError(f"Background file not found:\n{bg_file}")
    if not os.path.exists(filament_h5):
        raise FileNotFoundError(f"Filament file not found:\n{filament_h5}")

    # Load background RA/DEC
    with h5py.File(bg_file, "r") as f:
        ra_bg  = f["ra"][:]
        dec_bg = f["dec"][:]
        
    ra_min, ra_max = float(np.min(ra_bg)), float(np.max(ra_bg))
    dec_min, dec_max = float(np.min(dec_bg)), float(np.max(dec_bg))

    #Background density image

    bins_ra = 1400
    bins_dec = 900

    H, xedges, yedges = np.histogram2d(
        ra_bg, dec_bg,
        bins=[bins_ra, bins_dec],
        range=[[ra_min, ra_max], [dec_min, dec_max]],
    )
    H = np.log10(H + 1.0)

    #  Load filament points (RA/DEC/Label)

    with h5py.File(filament_h5, "r") as h:
        d = h[fil_dataset_name][:]
        ra_fil  = d["RA"]
        dec_fil = d["DEC"]
        lab     = d["Filament_Label"]

    # drop noise label if we use -1 
    keep = (lab >= 0)
    ra_fil, dec_fil, lab = ra_fil[keep], dec_fil[keep], lab[keep]

    #subsample filaments 
    max_fil_pts = 2_000_000
    if ra_fil.size > max_fil_pts:
        idx = np.random.default_rng(0).choice(ra_fil.size, size=max_fil_pts, replace=False)
        ra_fil, dec_fil, lab = ra_fil[idx], dec_fil[idx], lab[idx]

    # ----------------------------
    # Plot
    # ----------------------------
    out_png = os.path.join(out_dir, "background_density_with_filaments_DES_sim.png")

    plt.figure(figsize=(10, 6.5))
    plt.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[ra_min, ra_max, dec_min, dec_max],
    )

    #  overlap: all filament points as one layer
    plt.scatter(ra_fil, dec_fil, s=0.2, alpha=0.9)

    plt.xlabel("RA [deg]")
    plt.ylabel("DEC [deg]")
    plt.title("DES Y3 background density + filaments overlay")
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()

    print(f"[OK] Saved → {out_png}")


if __name__ == "__main__":
    main()

