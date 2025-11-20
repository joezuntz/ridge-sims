import os
import pandas as pd
import matplotlib.pyplot as plt

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_shear(input_dir="Cosmo_sim_ridges", output_dir=None):

    input_dir = os.path.abspath(input_dir)

    if output_dir is not None:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):

        if os.path.basename(root) != "shear":
            continue

        print(f"\nFound shear directory:\n  {root}")

        csv_files = [f for f in files if f.startswith("shear_p") and f.endswith(".csv")]
        if not csv_files:
            print("  (No shear_pXX.csv files found, skipping.)")
            continue

        gplus_data, gx_data = [], []
        labels_gplus, labels_gx = [], []

        # Read CSVs
        for fname in csv_files:
            path = os.path.join(root, fname)
            try:
                df = pd.read_csv(path)
                if "g_plus" in df.columns:
                    gplus_data.append(df["g_plus"].values)
                    labels_gplus.append(os.path.basename(root) + "_" + fname)

                if "g_x" in df.columns:
                    gx_data.append(df["g_x"].values)
                    labels_gx.append(os.path.basename(root) + "_" + fname)

            except Exception as e:
                print(f"  WARNING: Could not read {path} → {e}")
                continue

        # Always save in a single folder
        save_root = output_dir if output_dir else root
        os.makedirs(save_root, exist_ok=True)

        # g_plus
        if gplus_data:
            plt.figure(figsize=(8, 5))
            for data, label in zip(gplus_data, labels_gplus):
                plt.plot(data, label=label)
            plt.title("All g+ curves")
            plt.xlabel("Index")
            plt.ylabel("g+")
            plt.legend()
            out_file = os.path.join(save_root, "all_gplus.png")
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"  Saved: {out_file}")

        # g_x
        if gx_data:
            plt.figure(figsize=(8, 5))
            for data, label in zip(gx_data, labels_gx):
                plt.plot(data, label=label)
            plt.title("All g× curves")
            plt.xlabel("Index")
            plt.ylabel("g×")
            plt.legend()
            out_file = os.path.join(save_root, "all_gx.png")
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"  Saved: {out_file}")

    print("\nAll plotting completed.\n")

if __name__ == "__main__":
    plot_all_shear(
    input_dir="Cosmo_sim_ridges",
    output_dir="plots"
)
