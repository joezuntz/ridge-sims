import os
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt




## ============================================================
## PATHS (run from DESY3/)
## ============================================================
#current_dir = os.path.dirname(os.path.abspath(__file__))

#ridge_dir = os.path.join(current_dir, "DES_ridge_analysis", "Ridges_analysis")
#out_dir   = os.path.join(current_dir, "test_plots")
#os.makedirs(out_dir, exist_ok=True)

## ============================================================
## MAIN
## ============================================================
#def main():
#    if not os.path.isdir(ridge_dir):
#        raise FileNotFoundError(f"Ridge directory not found:\n{ridge_dir}")

#    ridge_files = sorted(glob.glob(os.path.join(ridge_dir, "*.h5")))
#    if len(ridge_files) == 0:
#        raise FileNotFoundError(f"No .h5 files found in:\n{ridge_dir}")

#    print(f"[INFO] Found {len(ridge_files)} ridge files.")
#    print(f"[INFO] Writing plots to: {out_dir}")

#    for h5_path in ridge_files:
#        base = os.path.basename(h5_path).replace(".h5", "")

#        with h5py.File(h5_path, "r") as f:
#            if "ridges" not in f:
#                print(f"[SKIP] No 'ridges' dataset in: {h5_path}")
#                continue
#            ridges = f["ridges"][:]

#        # Plot EXACTLY as stored:
#        # x = ridges[:,1], y = ridges[:,0]
#        out_png = os.path.join(out_dir, f"{base}_raw.png")

#        plt.figure(figsize=(8, 6))
#        plt.scatter(ridges[:, 1], ridges[:, 0], s=0.4, alpha=0.6)
#        plt.xlabel("col 1 (as stored)")
#        plt.ylabel("col 0 (as stored)")
#        plt.title(base)
#        plt.grid(True, which="both", ls="--", alpha=0.3)
#        plt.tight_layout()
#        plt.savefig(out_png, dpi=200)
#        plt.close()

#        print(f"[OK] {out_png}")

#    print("[DONE]")

#if __name__ == "__main__":
#    main()




## ============================================================
## PATHS (run from DESY3/)
## ============================================================
#current_dir = os.path.dirname(os.path.abspath(__file__))

#fil_dir = os.path.join(current_dir, "filaments")
#out_dir = os.path.join(current_dir, "test_plots")
#os.makedirs(out_dir, exist_ok=True)

## ============================================================
## MAIN
## ============================================================
#def main():
#    if not os.path.isdir(fil_dir):
#        raise FileNotFoundError(f"Filament directory not found:\n{fil_dir}")

#    fil_files = sorted(glob.glob(os.path.join(fil_dir, "*.h5")))
#    if len(fil_files) == 0:
#        raise FileNotFoundError(f"No filament .h5 files found in:\n{fil_dir}")

#    print(f"[INFO] Found {len(fil_files)} filament files.")
#    print(f"[INFO] Writing plots to: {out_dir}")

#    for h5_path in fil_files:
#        base = os.path.basename(h5_path).replace(".h5", "")

#        with h5py.File(h5_path, "r") as f:
#            if "data" not in f:
#                print(f"[SKIP] No 'data' dataset in: {h5_path}")
#                continue

#            dset = f["data"][:]

#        # Expected structured dtype: ("RA","DEC","Filament_Label")
#        # Plot EXACTLY as stored.
#        try:
#            ra = dset["RA"]
#            dec = dset["DEC"]
#            lab = dset["Filament_Label"]
#        except Exception as e:
#            raise RuntimeError(
#                f"{h5_path} does not look like the expected structured dataset "
#                f"with fields RA/DEC/Filament_Label. Got dtype={dset.dtype}"
#            ) from e

#        out_png = os.path.join(out_dir, f"{base}_filaments_raw.png")

#        plt.figure(figsize=(8, 6))

#        # Color by label; noise label -1 will be a separate color automatically
#        sc = plt.scatter(ra, dec, c=lab, s=0.4, alpha=0.7)

#        plt.xlabel("RA (as stored)")
#        plt.ylabel("DEC (as stored)")
#        plt.title(base)
#        plt.grid(True, which="both", ls="--", alpha=0.3)

#        # Colorbar helps confirm labels are sane
#        cb = plt.colorbar(sc)
#        cb.set_label("Filament_Label")

#        plt.tight_layout()
#        plt.savefig(out_png, dpi=200)
#        plt.close()

#        print(f"[OK] {out_png}")

#    print("[DONE]")

#if __name__ == "__main__":
#    main()



# ============================================================
# PATHS (run from DESY3/)
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))

base_catalog_dir = os.path.join(parent_dir, "des-data")
bg_file = os.path.join(
    base_catalog_dir,
    "des-y3-ridges-background-v2_cutzl0.40.h5"
)

out_dir = os.path.join(current_dir, "test_plots")
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# MAIN
# ============================================================
def main():

    if not os.path.exists(bg_file):
        raise FileNotFoundError(f"Background file not found:\n{bg_file}")

    with h5py.File(bg_file, "r") as f:
        ra  = f["ra"][:]      # as stored
        dec = f["dec"][:]     # as stored

        # optional extras if you want to sanity-check
        g1 = f["g1"][:]
        g2 = f["g2"][:]
        z  = f["z"][:]

    print(f"[INFO] Loaded background:")
    print(f"       N = {len(ra)}")
    print(f"       ra range  = [{ra.min()}, {ra.max()}]")
    print(f"       dec range = [{dec.min()}, {dec.max()}]")
    print(f"       z range   = [{z.min()}, {z.max()}]")

    out_png = os.path.join(out_dir, "background_raw.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(ra, dec, s=0.2, alpha=0.4)
    plt.xlabel("ra (as stored)")
    plt.ylabel("dec (as stored)")
    plt.title("DES Y3 background (raw)")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[OK] Saved plot → {out_png}")


if __name__ == "__main__":
    main()