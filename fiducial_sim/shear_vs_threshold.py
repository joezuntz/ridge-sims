import os, sys
import re
import glob

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
#  discovery Function
# ===============================================================
def find_contracted_files(root_dir):
    contracted = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith("_contracted.h5"):
                contracted.append(os.path.join(root, f))
    return sorted(contracted)


# ===============================================================
#   Extract threshold and run_id from path
# ===============================================================
def extract_fp_from_path(path):
    m = re.search(r"Ridges_final_p(\d+)", path)
    if m:
        return int(m.group(1))

    m = re.search(r"_ridges_p(\d+)_contracted\.h5$", os.path.basename(path))
    if m:
        return int(m.group(1))

    return None


def extract_runid_from_path(path):
    # matches ".../run_1_mesh_2_band_0.1/..."
    m = re.search(r"(?:^|/)run_(\d+)_mesh_", path)
    return int(m.group(1)) if m else None


def extract_band_from_path(path):
    # matches ".../run_1_mesh_2_band_0.1/..."
    m = re.search(r"(?:^|/)run_\d+_mesh_\d+_band_([0-9.]+)(?:/|$)", path)
    return float(m.group(1)) if m else None


def extract_mesh_from_path(path):
    # matches ".../run_1_mesh_2_band_0.1/..."
    m = re.search(r"(?:^|/)run_\d+_mesh_(\d+)_band_[0-9.]+(?:/|$)", path)
    return float(m.group(1)) if m else None


# ===============================================================
# ============================ MAIN =============================
# ===============================================================
def main():

    
    output_base = "density_threshold_test"

    # shear outputs
    out_root = os.path.join(output_base, "shear_vs_threshold")
    if rank == 0:
        os.makedirs(out_root, exist_ok=True)
    comm.Barrier()

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
    comm.Barrier()

    # -----------------------------------------------------------
    # Noise catalogs 
    # -----------------------------------------------------------
    n_random_rotations = 50
    BACKGROUND_TYPE_NOISE = "noise"

    # -----------------------------------------------------------
    # Discover contracted ridge files
    # -----------------------------------------------------------
    contracted_files = None
    if rank == 0:
        contracted_files = find_contracted_files(output_base)
        print(f"Found {len(contracted_files)} contracted ridge files under {output_base}\n")
    contracted_files = comm.bcast(contracted_files, root=0)

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
                if rank == 0:
                    skipped_missing_h5.append(h5_file)
                    print(f"[missing] {h5_file}")
                continue

            band = extract_band_from_path(h5_file)
            mesh = extract_mesh_from_path(h5_file)

            if (band is None) or (mesh is None):
                if rank == 0:
                    skipped_wrong_tree.append(h5_file)
                    print(f"[skip] Could not parse band/mesh from path: {h5_file}")
                continue

            if (abs(band - REQUIRED_BAND) > 1e-9) or (abs(mesh - REQUIRED_MESH) > 1e-9):
                if rank == 0:
                    skipped_wrong_tree.append(h5_file)
                    print(f"[skip] Not band={REQUIRED_BAND} mesh={REQUIRED_MESH}: {h5_file}")
                continue

            fp = extract_fp_from_path(h5_file)
            if fp is None:
                if rank == 0:
                    skipped_missing_fp.append(h5_file)
                    print(f"[skip] Could not extract fp: {h5_file}")
                continue

            run_id = extract_runid_from_path(h5_file)
            run_tag = f"run_{run_id}_" if run_id is not None else ""

            # Output files (signal)
            shear_csv    = os.path.join(out_root, f"{run_tag}shear_fp_{fp:02d}.csv")
            filaments_h5 = os.path.join(out_root, f"{run_tag}filaments_fp_{fp:02d}.h5")

            # Random shear directory for their corresponding density threshold
            random_dir = os.path.join(out_root, f"{run_tag}random_shear_p{fp:02d}")
            if rank == 0:
                os.makedirs(random_dir, exist_ok=True)
            comm.Barrier()

            # ---------------------------------------------------
            # SIGNAL SHEAR + FILAMENTS
            # ---------------------------------------------------
            signal_exists = True
            if rank == 0:
                signal_exists = os.path.exists(shear_csv) and os.path.exists(filaments_h5)
            signal_exists = comm.bcast(signal_exists, root=0)
            
            if signal_exists:
                if rank == 0:
                    skipped_existing_output.append(shear_csv)
                    print(f"[skip] output exists: {shear_csv} (+ filaments)")
            else:
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


            comm.Barrier()

            # ---------------------------------------------------
            # NOISE SHEAR 
            # ---------------------------------------------------

            noise_dir = os.path.join(parent_dir, "DES_sim/DES_sim/noise")
            noise_files = []
            if rank == 0:
                noise_files = sorted(glob.glob(os.path.join(noise_dir, "source_catalog_noise_*.h5")))
                noise_files = noise_files[:n_random_rotations]
                if len(noise_files) == 0:
                    print(f"[WARN] No noise files found in:\n  {noise_dir}")
            noise_files = comm.bcast(noise_files, root=0)

            for i, nf in enumerate(noise_files):
                random_csv = os.path.join(random_dir, f"shear_random_{i:03d}.csv")

                exists = True
                if rank == 0:
                    exists = os.path.exists(random_csv)
                exists = comm.bcast(exists, root=0)
                if exists:
                    continue

                process_shear_sims(
                    filament_file=filaments_h5,
                    bg_data=nf,
                    output_shear_file=random_csv,
                    k=1, num_bins=20, comm=comm,
                    flip_g1=False, flip_g2=False,
                    background_type=BACKGROUND_TYPE_NOISE,
                    nside_coverage=32,
                    min_distance_arcmin=1.0,
                    max_distance_arcmin=60.0
                )
                comm.Barrier()

            if rank == 0:
                print(f"[done] fp={fp:02d} noise → {random_dir}")

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
