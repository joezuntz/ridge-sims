import os
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt







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







# PATHS

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))

bg_file = os.path.join(parent_dir, "des-data", "des-y3-ridges-background-v2_cutzl0.40.h5")

# Filaments 
filament_h5 = os.path.join(current_dir, "filaments", "filaments_p15.h5")
fil_dataset_name = "data"  

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
    out_png = os.path.join(out_dir, "background_density_with_filaments.png")

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

