"""
Runs shear computation for all runs and bands, alternating between
normal and zero-error directories for each band.

Folder structure created inside each band:
band_X/shear_calc/
    ├── raw_ridges_shear/
    │   ├── filament_segments/
    │   └── noise_shear/
    └── shrinked_ridges_shear/
        ├── filament_segments/
        └── noise_shear/
"""

import os
import h5py
import numpy as np
import time
from ridge_analysis_tools import *   # assumes these functions exist
from mpi4py import MPI

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# === CONFIGURATION ===
bands = ["band_0.1"]           # adjust as needed
run_ids = [1]                  # adjust as needed
final_percentiles = [15]       # fixed to 15 in your code
base_root = "simulation_ridges_comparative_analysis"
noise_dir = "example_zl04_mesh5e5/noise"

# background mapping per variant
sim_bg_map = {
    "normal": "lhc_run_sims",
    "zero_err": "lhc_run_sims_zero_err_10"
}

# Collect all noise realizations (rank 0)
if rank == 0:
    noise_files = sorted(
        [f for f in os.listdir(noise_dir) if f.startswith("source_catalog_noise_") and f.endswith(".h5")],
        key=lambda x: int(x.split("_")[-1].replace(".h5", ""))
    )
else:
    noise_files = None
# broadcast list to all ranks
noise_files = comm.bcast(noise_files, root=0)


# ======================================================
# === Compute shear for a given ridge file (safe I/O) ===
# ======================================================
def compute_shear_for_ridges(ridges_h5, bg_data, output_dir):
    """
    Safe wrapper: rank 0 performs all calls that write files (save_filaments_to_hdf5,
    process_shear_sims, np.savetxt). Other ranks wait on Barriers.
    """
    # Make output directories (everyone can create safely)
    filament_dir = os.path.join(output_dir, "filament_segments")
    noise_shear_dir = os.path.join(output_dir, "noise_shear")
    if rank == 0:
        os.makedirs(filament_dir, exist_ok=True)
        os.makedirs(noise_shear_dir, exist_ok=True)
    comm.Barrier()

    fp = final_percentiles[0] if len(final_percentiles) == 1 else final_percentiles[0]

    # Load ridges (safe to read on all ranks)
    try:
        with h5py.File(ridges_h5, "r") as f:
            ridges = f["ridges"][:]
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Could not read ridges file {ridges_h5}: {e}")
        return

    # Build filaments & labels on all ranks (computation-only)
    mst = build_mst(ridges)
    branch_points = detect_branch_points(mst)
    filament_segments = split_mst_at_branches(mst, branch_points)
    filament_labels = segment_filaments_with_dbscan(ridges, filament_segments)

    filament_h5 = os.path.join(filament_dir, f"filaments_p{fp:02d}.h5")

    # Save filaments -> ONLY rank 0 (to avoid locking)
    if rank == 0:
        try:
            save_filaments_to_hdf5(ridges, filament_labels, filament_h5)
            print(f"[rank 0] Saved filaments -> {filament_h5}")
        except Exception as e:
            print(f"[rank 0][ERROR] save_filaments_to_hdf5 failed: {e}")
            # still continue; but downstream will likely fail
    comm.Barrier()

    # --- signal shear: run ONLY on rank 0 ---
    shear_csv = os.path.join(filament_dir, f"shear_p{fp:02d}.csv")
    shear_flip_csv = os.path.join(filament_dir, f"shear_p{fp:02d}_flipG1.csv")

    if rank == 0:
        try:
            process_shear_sims(filament_h5, bg_data,
                               output_shear_file=shear_csv,
                               background_type='sim', plot_output_dir=filament_dir)
            process_shear_sims(filament_h5, bg_data,
                               output_shear_file=shear_flip_csv,
                               flip_g1=True, background_type='sim')
            print(f"[rank 0] Computed signal shear -> {shear_csv} (+ flipped)")
        except Exception as e:
            print(f"[rank 0][ERROR] process_shear_sims(signal) failed for {filament_h5}: {e}")
    comm.Barrier()

    # === noise shears: run ONLY on rank 0 ===
    all_noise_profiles = []
    all_noise_flip_profiles = []
    if rank == 0:
        for nf in noise_files:
            try:
                realization_id = nf.split("_")[-1].replace(".h5", "")
                noise_file = os.path.join(noise_dir, nf)
                shear_noise_csv_i = os.path.join(noise_shear_dir, f"shear_noise_p{fp:02d}_{realization_id}.csv")
                shear_noise_flip_csv_i = os.path.join(noise_shear_dir, f"shear_noise_p{fp:02d}_{realization_id}_flipG1.csv")

                process_shear_sims(filament_h5, noise_file, output_shear_file=shear_noise_csv_i, background_type='sim')
                process_shear_sims(filament_h5, noise_file, output_shear_file=shear_noise_flip_csv_i, flip_g1=True, background_type='sim')

                # Load results
                dat = np.loadtxt(shear_noise_csv_i, delimiter=",", skiprows=1)
                datf = np.loadtxt(shear_noise_flip_csv_i, delimiter=",", skiprows=1)
                all_noise_profiles.append(dat)
                all_noise_flip_profiles.append(datf)
            except Exception as e:
                print(f"[rank 0][WARN] Failed for noise {nf}: {e}")

        # convert to arrays and compute mean if possible
        if len(all_noise_profiles) == 0:
            print("[rank 0][WARN] No noise profiles produced; skipping mean subtraction.")
            mean_noise = mean_noise_flip = None
        else:
            all_noise_profiles = np.array(all_noise_profiles)
            all_noise_flip_profiles = np.array(all_noise_flip_profiles)
            mean_noise = np.mean(all_noise_profiles, axis=0)
            mean_noise_flip = np.mean(all_noise_flip_profiles, axis=0)
    else:
        mean_noise = None
        mean_noise_flip = None

    # Broadcast flag whether mean_noise is available
    has_mean = comm.bcast(mean_noise is not None, root=0)

    # If rank 0 computed mean_noise, we proceed to subtract / save (rank 0 writes)
    if has_mean:
        if rank == 0:
            try:
                shear_data = np.loadtxt(shear_csv, delimiter=",", skiprows=1)
                shear_data_flip = np.loadtxt(shear_flip_csv, delimiter=",", skiprows=1)

                g_plus_subtracted = shear_data[:, 2] - mean_noise[:, 2]
                g_cross_subtracted = shear_data[:, 3] - mean_noise[:, 3]
                subtracted_data = np.column_stack((shear_data[:, 0], shear_data[:, 1],
                                                   g_plus_subtracted, g_cross_subtracted,
                                                   shear_data[:, 4], shear_data[:, 5]))
                out_file = os.path.join(filament_dir, f"shear_p{fp:02d}_shear-randomshear.csv")
                header = "Bin_Center,Weighted_Real_Distance,Weighted_g_plus_subtracted,Weighted_g_cross_subtracted,Counts,bin_weight"
                np.savetxt(out_file, subtracted_data, delimiter=",", header=header, comments="")
                print(f"[rank 0] Saved mean-noise-subtracted shear -> {out_file}")

                # flipped
                g_plus_subtracted_flip = shear_data_flip[:, 2] - mean_noise_flip[:, 2]
                g_cross_subtracted_flip = shear_data_flip[:, 3] - mean_noise_flip[:, 3]
                out_flip = np.column_stack((shear_data_flip[:, 0], shear_data_flip[:, 1],
                                            g_plus_subtracted_flip, g_cross_subtracted_flip,
                                            shear_data_flip[:, 4], shear_data_flip[:, 5]))
                out_file_flip = os.path.join(filament_dir, f"shear_p{fp:02d}_flipG1_shear-randomshear.csv")
                np.savetxt(out_file_flip, out_flip, delimiter=",", header=header, comments="")
                print(f"[rank 0] Saved flipped mean-noise-subtracted shear -> {out_file_flip}")
            except Exception as e:
                print(f"[rank 0][ERROR] Failed to subtract/save mean noise: {e}")
    comm.Barrier()

    if rank == 0:
        print(f"[rank 0] Finished shear pipeline for {os.path.basename(ridges_h5)}")


# ================================================
# === MAIN LOOP ===
# ================================================
def main():
    for band in bands:
        for run_id in run_ids:
            # process normal then zero_err (interleaved as requested)
            for variant in ["normal", "zero_err"]:
                if rank == 0:
                    print(f"\n--- Running {band}/{variant}/run_{run_id} ---")

                base_dir = os.path.join(base_root, variant, band)
                shear_calc_dir = os.path.join(base_dir, "shear_calc")
                if rank == 0:
                    os.makedirs(shear_calc_dir, exist_ok=True)
                comm.Barrier()

                # determine background catalog path for this run
                sim_base = sim_bg_map.get(variant)
                if sim_base is None:
                    if rank == 0:
                        print(f"[WARN] Unknown variant mapping for {variant} – skipping")
                    continue
                bg_data = os.path.join(sim_base, f"run_{run_id}", "source_catalog_cutzl04.h5")
                if rank == 0 and not os.path.exists(bg_data):
                    print(f"[WARN] Background file not found: {bg_data} (will still attempt if present)")

                # loop over raw / shrinked
                for ridge_type in ["raw_ridges", "shrinked_ridges"]:
                    ridge_subdir = "Ridges_final_p15" if ridge_type == "raw_ridges" else "Shrinked_Ridges_final_p15"
                    # filenames include variant and run id:
                    if ridge_type == "raw_ridges":
                        ridge_filename = f"{variant}_run_{run_id}_ridges_p15.h5"
                    else:
                        ridge_filename = f"{variant}_run_{run_id}_ridges_p15_shrinked.h5"

                    ridge_h5 = os.path.join(base_dir, ridge_subdir, ridge_filename)

                    if rank == 0:
                        if not os.path.exists(ridge_h5):
                            print(f"[WARN] Ridge file not found: {ridge_h5} - skipping {ridge_type}")
                            found = False
                        else:
                            found = True
                    else:
                        found = None
                    # broadcast presence
                    found = comm.bcast(found, root=0)
                    if not found:
                        continue

                    output_dir = os.path.join(shear_calc_dir, f"{ridge_type}_shear")
                    # call compute (internally rank0 will write and other ranks will wait)
                    compute_shear_for_ridges(ridge_h5, bg_data, output_dir)

    if rank == 0:
        print("\nAll processing complete.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    comm.Barrier()
    if rank == 0:
        print(f"Total runtime: {time.time() - t0:.1f} s")