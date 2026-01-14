import os, sys
import re

# ===============================================================
# PATH SETUP
# ===============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

os.chdir(current_dir)

import numpy as np
from mpi4py import MPI

from ridge_analysis_tools import *

comm = MPI.COMM_WORLD
rank = comm.rank


# ===============================================================
#   Structured discovery
# ===============================================================
def find_contracted_files_parameter_test(root="parameter_test"):
    """
    Expected structure:

    parameter_test/
      run_*/
        band_*/
          mesh_*/
            Ridges_final_pXX/
              *_contracted.h5
    """
    contracted = []

    if not os.path.exists(root):
        return contracted

    for run_dir in sorted(os.listdir(root)):
        run_path = os.path.join(root, run_dir)
        if (not run_dir.startswith("run_")) or (not os.path.isdir(run_path)):
            continue

        for band_dir in sorted(os.listdir(run_path)):
            band_path = os.path.join(run_path, band_dir)
            if (not band_dir.startswith("band_")) or (not os.path.isdir(band_path)):
                continue

            for mesh_dir in sorted(os.listdir(band_path)):
                mesh_path = os.path.join(band_path, mesh_dir)
                if (not mesh_dir.startswith("mesh_")) or (not os.path.isdir(mesh_path)):
                    continue

                for sub in sorted(os.listdir(mesh_path)):
                    if not sub.startswith("Ridges_final_p"):
                        continue

                    ridges_path = os.path.join(mesh_path, sub)
                    if not os.path.isdir(ridges_path):
                        continue

                    for f in os.listdir(ridges_path):
                        if f.endswith("_contracted.h5"):
                            contracted.append(os.path.join(ridges_path, f))

    return sorted(contracted)


# ===============================================================
#   Extract fp and run_id from path
# ===============================================================
def extract_fp_from_path(path):
    """
    Prefer directory tag:
      .../Ridges_final_p15/...
    fallback to filename:
      ..._ridges_p15_contracted.h5
    """
    m = re.search(r"Ridges_final_p(\d+)", path)
    if m:
        return int(m.group(1))

    m = re.search(r"_ridges_p(\d+)_contracted\.h5$", os.path.basename(path))
    if m:
        return int(m.group(1))

    return None


def extract_runid_from_path(path):
    m = re.search(r"(?:^|/)run_(\d+)(?:/|$)", path)
    return int(m.group(1)) if m else None


def extract_band_from_path(path):
    m = re.search(r"(?:^|/)band_([0-9.]+)(?:/|$)", path)
    return float(m.group(1)) if m else None


def extract_mesh_from_path(path):
    m = re.search(r"(?:^|/)mesh_([0-9.]+)(?:/|$)", path)
    return float(m.group(1)) if m else None


# ===============================================================
# ============================ MAIN =============================
# ===============================================================
def main():

    parameter_root = "parameter_test"

    # Where shear go
    out_root = os.path.join(parameter_root, "shear_vs_fp")
    if rank == 0:
        os.makedirs(out_root, exist_ok=True)

    # -----------------------------------------------------------
    # constraints 
    # -----------------------------------------------------------
    REQUIRED_BAND = 0.1
    REQUIRED_MESH = 2.0

    # -----------------------------------------------------------
    # Background catalog 
    # -----------------------------------------------------------
    BG_data = os.path.join(
        parent_dir,
        "lhc_run_sims_zero_err_10",
        "run_1",
        "source_catalog_cutzl04.h5"
    )

    if rank == 0 and (not os.path.exists(BG_data)):
        print(f"[FATAL] Missing background file:\n  {BG_data}")
        return

    # -----------------------------------------------------------
    # Discover contracted ridge files
    # -----------------------------------------------------------
    contracted_files = find_contracted_files_parameter_test(parameter_root)

    if rank == 0:
        print(f"Found {len(contracted_files)} contracted ridge files under {parameter_root}\n")

    skipped_missing_h5 = []
    skipped_missing_fp = []
    skipped_wrong_tree = []
    skipped_existing_output = []

    # -----------------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------------
    for h5_file in contracted_files:

        try:
            if not os.path.exists(h5_file):
                skipped_missing_h5.append(h5_file)
                if rank == 0:
                    print(f"[missing] {h5_file}")
                continue

            # fp-test tree:
            band = extract_band_from_path(h5_file)
            mesh = extract_mesh_from_path(h5_file)

            # If we can't parse these, treat as wrong tree 
            if (band is None) or (mesh is None):
                skipped_wrong_tree.append(h5_file)
                if rank == 0:
                    print(f"[skip] Could not parse band/mesh from path: {h5_file}")
                continue

            # numeric check
            if (abs(band - REQUIRED_BAND) > 1e-9) or (abs(mesh - REQUIRED_MESH) > 1e-9):
                skipped_wrong_tree.append(h5_file)
                if rank == 0:
                    print(f"[skip] Not band={REQUIRED_BAND} mesh={REQUIRED_MESH}: {h5_file}")
                continue

            # Extract fp 
            fp = extract_fp_from_path(h5_file)
            if fp is None:
                skipped_missing_fp.append(h5_file)
                if rank == 0:
                    print(f"[skip] Could not extract fp: {h5_file}")
                continue

            run_id = extract_runid_from_path(h5_file)
            run_tag = f"run_{run_id}_" if run_id is not None else ""

            # Output files
            shear_csv    = os.path.join(out_root, f"{run_tag}shear_fp_{fp:02d}.csv")
            filaments_h5 = os.path.join(out_root, f"{run_tag}filaments_fp_{fp:02d}.h5")

            # Skip existing output
            if os.path.exists(shear_csv):
                skipped_existing_output.append(shear_csv)
                if rank == 0:
                    print(f"[skip] output exists: {shear_csv}")
                continue

            # Run shear computation
            process_ridge_file(
                h5_file=h5_file,
                BG_data=BG_data,
                filament_h5=filaments_h5,
                shear_csv=shear_csv,
                background_type="sim",    
                shear_flip_csv=None,
                comm=comm
            )

            if rank == 0:
                print(f"[done] fp={fp:02d} → {shear_csv}")

        except Exception as e:
            if rank == 0:
                print(f"[ERROR] {h5_file} → {e}")

    # ===========================================================
    # FINAL SUMMARY
    # ===========================================================
    if rank == 0:
        print("\n================== FINAL SUMMARY ==================\n")

        if skipped_missing_h5:
            print(f"Missing contracted files ({len(skipped_missing_h5)}):")
            for f in skipped_missing_h5:
                print("  -", f)

        if skipped_wrong_tree:
            print(f"\nSkipped (wrong/unparseable band/mesh tree) ({len(skipped_wrong_tree)}):")
            for f in skipped_wrong_tree:
                print("  -", f)

        if skipped_missing_fp:
            print(f"\nCould not parse fp ({len(skipped_missing_fp)}):")
            for f in skipped_missing_fp:
                print("  -", f)

        if skipped_existing_output:
            print(f"\nSkipped existing outputs ({len(skipped_existing_output)}):")
            for f in skipped_existing_output:
                print("  -", f)

        if not (skipped_missing_h5 or skipped_wrong_tree or skipped_missing_fp or skipped_existing_output):
            print("No files were skipped.")

        print("\n==================================================")
        print("All shear calculations completed.")


if __name__ == "__main__":
    main()
