import os
import glob
import h5py
import matplotlib.pyplot as plt

# ============================================================
# PATHS (run from DESY3/)
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))

ridge_dir = os.path.join(current_dir, "DES_ridge_analysis", "Ridges_analysis")
out_dir   = os.path.join(current_dir, "test_plots")
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.isdir(ridge_dir):
        raise FileNotFoundError(f"Ridge directory not found:\n{ridge_dir}")

    ridge_files = sorted(glob.glob(os.path.join(ridge_dir, "*.h5")))
    if len(ridge_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in:\n{ridge_dir}")

    print(f"[INFO] Found {len(ridge_files)} ridge files.")
    print(f"[INFO] Writing plots to: {out_dir}")

    for h5_path in ridge_files:
        base = os.path.basename(h5_path).replace(".h5", "")

        with h5py.File(h5_path, "r") as f:
            if "ridges" not in f:
                print(f"[SKIP] No 'ridges' dataset in: {h5_path}")
                continue
            ridges = f["ridges"][:]

        # Plot EXACTLY as stored:
        # x = ridges[:,1], y = ridges[:,0]
        out_png = os.path.join(out_dir, f"{base}_raw.png")

        plt.figure(figsize=(8, 6))
        plt.scatter(ridges[:, 1], ridges[:, 0], s=0.4, alpha=0.6)
        plt.xlabel("col 1 (as stored)")
        plt.ylabel("col 0 (as stored)")
        plt.title(base)
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        print(f"[OK] {out_png}")

    print("[DONE]")

if __name__ == "__main__":
    main()
