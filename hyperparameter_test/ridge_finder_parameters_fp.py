import os, sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpi4py import MPI

# Path setup --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

os.chdir(current_dir)

import dredge_scms
from ridge_analysis_tools import (
    load_coordinates,
    build_density_map,
    redo_cuts,
    results_plot,
    ridge_edge_filter_disk,
)

COMM = MPI.COMM_WORLD
RANK = COMM.rank


# ------------------------------------------------------------
# Local contraction helper (rank 0 only)
# ------------------------------------------------------------
def process_ridge_file_local(ridge_file, mask, nside, radius_arcmin, min_coverage, output_dir, plot_dir):
    """Apply the disk-coverage filter to one ridge file and write *_contracted.h5 + diagnostic scatter."""
    with h5py.File(ridge_file, "r") as f:
        ridges = f["ridges"][:]

    ridge_dec = ridges[:, 0]
    ridge_ra  = ridges[:, 1]
    n_total = len(ridges)

    keep_idx = ridge_edge_filter_disk(
        ridge_ra, ridge_dec, mask, nside,
        radius_arcmin=radius_arcmin, min_coverage=min_coverage
    )
    ridges_clean = ridges[keep_idx]
    print(f"[contracted] {os.path.basename(ridge_file)}: kept {len(ridges_clean)}/{n_total}")

    # Save contracted ridges next to original
    base_name = os.path.basename(ridge_file).replace(".h5", "_contracted.h5")
    out_file = os.path.join(output_dir, base_name)
    with h5py.File(out_file, "w") as f:
        f.create_dataset("ridges", data=ridges_clean)

    # Diagnostic scatter
    plot_file = os.path.join(plot_dir, os.path.basename(out_file).replace(".h5", "_diagnostic.png"))
    plt.figure(figsize=(8, 6))
    plt.scatter(ridge_ra, ridge_dec, s=1, alpha=0.3, label="All ridges")
    plt.scatter(ridges_clean[:, 1], ridges_clean[:, 0], s=1, alpha=0.6, label="Filtered ridges")
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title(f"contracted ridges\nradius={radius_arcmin} arcmin, min_cov={min_coverage}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    plt.close()

    print(f"[plot] Saved diagnostic → {plot_file}")
    return out_file


# ==============================================================
# MAIN SCRIPT
# ==============================================================
def main():

    # ----------------------------------------------------------
    # Fixed parameters
    # ----------------------------------------------------------
    mesh_N = 2
    bandwidth = 0.1
    final_percentiles = [15] #[10, 25, 40, 55, 70]   
    run_ids = range(1, 2)  

    # Filament finder params (match your ridge_analysis_tools defaults)
    neighbours = 5000
    convergence = 1e-5
    seed = 3482364
    mesh_size = int(mesh_N * 5e5)

    sim_bases = {
        "zero_err": "lhc_run_sims_zero_err_10",
    }

    # Output root
    output_base = "parameter_test"
    if RANK == 0:
        os.makedirs(output_base, exist_ok=True)

    # Contraction parameters
    radius_arcmin = 4.0
    min_coverage = 0.9
    nside = 512
    mask_filename = os.path.join(parent_dir, "des-data", "desy3_gold_mask.npy")
    mask = np.load(mask_filename) if RANK == 0 else None
    mask = COMM.bcast(mask, root=0)

    # ----------------------------------------------------------
    # Summary bookkeeping
    # ----------------------------------------------------------
    summary = {
        "completed_raw": [],
        "loaded_raw_cache": [],
        "errors_raw": [],

        "completed_fp_outputs": [],
        "skipped_existing_fp_outputs": [],
        "errors_fp_outputs": [],

        "completed_contraction": [],
        "skipped_existing_contraction": [],
        "errors_contraction": [],
    }

    # ----------------------------------------------------------
    # Loop sims
    # ----------------------------------------------------------
    for base_label, base_folder in sim_bases.items():

        base_sim_dir = os.path.join(parent_dir, base_folder)

        for run_id in run_ids:

            # --------------------------------------------------
            # Directory layout (fixed mesh + fixed band)
            # --------------------------------------------------
            home_dir = os.path.join(
                output_base,
                f"run_{run_id}",
                f"band_{bandwidth:.1f}",
                f"mesh_{mesh_N}"
            )

            checkpoint_dir = os.path.join(home_dir, "checkpoints")
            plots_dir      = os.path.join(home_dir, "plots_by_final_percentile")

            if RANK == 0:
                os.makedirs(home_dir, exist_ok=True)
                os.makedirs(checkpoint_dir, exist_ok=True)
                os.makedirs(plots_dir, exist_ok=True)

            COMM.barrier()

            # --------------------------------------------------
            # Raw ridge 
            # --------------------------------------------------
            raw_file = os.path.join(home_dir, f"{base_label}_run_{run_id}_ridges_raw.h5")

            # Decide whether to run find_filaments
            raw_exists = True
            if RANK == 0:
                raw_exists = os.path.exists(raw_file)
            raw_exists = COMM.bcast(raw_exists, root=0)

            # ==================================================
            # Stage 1a: produce or load raw ridges + densities
            # ==================================================
            ridges = None
            initial_density = None
            final_density = None

            if raw_exists:
                # Load raw output (rank 0 then broadcast)
                if RANK == 0:
                    with h5py.File(raw_file, "r") as f:
                        ridges = f["ridges"][:]
                        initial_density = f["initial_density"][:]
                        final_density = f["final_density"][:]
                    summary["loaded_raw_cache"].append({"run": run_id})
                    print(f"[rank 0] Loaded raw cache → {raw_file}")

                ridges = COMM.bcast(ridges, root=0)
                initial_density = COMM.bcast(initial_density, root=0)
                final_density = COMM.bcast(final_density, root=0)

            else:
                # Compute coordinates once (rank 0) and broadcast
                coordinates = None
                if RANK == 0:
                    try:
                        coordinates = load_coordinates(base_sim_dir, run_id, shift=True, z_cut=0.4, fraction=None)
                    except Exception as e:
                        summary["errors_raw"].append({"run": run_id, "error": str(e)})
                        coordinates = None

                coordinates = COMM.bcast(coordinates, root=0)
                if coordinates is None:
                    # Can't proceed for this run
                    COMM.barrier()
                    continue

                # run once
                try:
                    ridges, initial_density, final_density = dredge_scms.find_filaments(
                        coordinates,
                        bandwidth=np.radians(bandwidth),
                        convergence=np.radians(convergence),
                        distance_metric="haversine",
                        n_neighbors=neighbours,
                        comm=COMM,
                        checkpoint_dir=checkpoint_dir,
                        resume=True,
                        seed=seed,
                        mesh_size=mesh_size,
                    )
                except Exception as e:
                    if RANK == 0:
                        summary["errors_raw"].append({"run": run_id, "error": str(e)})
                    COMM.barrier()
                    continue

                COMM.barrier()

                # Save raw (rank 0)
                if RANK == 0:
                    with h5py.File(raw_file, "w") as f:
                        f.create_dataset("ridges", data=ridges)
                        f.create_dataset("initial_density", data=initial_density)
                        f.create_dataset("final_density", data=final_density)
                        f.attrs["bandwidth_deg"] = float(bandwidth)
                        f.attrs["mesh_N"] = float(mesh_N)
                        f.attrs["mesh_size"] = int(mesh_size)
                        f.attrs["run_id"] = int(run_id)
                        f.attrs["base_label"] = str(base_label)
                    summary["completed_raw"].append({"run": run_id})
                    print(f"[rank 0] Saved raw ridges+density → {raw_file}")

            COMM.barrier()

            # ==================================================
            # Stage 1b: apply redo_cuts for each fp and write outputs
            # ==================================================
            # Build density map once reused for plots
            density_map = None
            if RANK == 0:
                try:
                    density_map = build_density_map(base_sim_dir, run_id, 512, z_cut=0.4)
                except Exception as e:
                    summary["errors_fp_outputs"].append({"run": run_id, "error": f"density_map: {e}"})
                    density_map = None

            for fp in final_percentiles:

                ridges_dir_fp = os.path.join(home_dir, f"Ridges_final_p{fp:02d}")
                if RANK == 0:
                    os.makedirs(ridges_dir_fp, exist_ok=True)

                ridge_file = os.path.join(
                    ridges_dir_fp,
                    f"{base_label}_run_{run_id}_ridges_p{fp:02d}.h5"
                )
                plot_file = os.path.join(
                    plots_dir,
                    f"{base_label}_run_{run_id}_Ridges_plot_p{fp:02d}.png"
                )

                # Skip if already exists
                fp_exists = True
                if RANK == 0:
                    fp_exists = os.path.exists(ridge_file) and os.path.exists(plot_file)
                    if fp_exists:
                        summary["skipped_existing_fp_outputs"].append({"run": run_id, "fp": fp})
                fp_exists = COMM.bcast(fp_exists, root=0)
                if fp_exists:
                    continue

                # Apply cut + save/plot (rank 0)
                if RANK == 0:
                    try:
                        ridges_cut = redo_cuts(
                            ridges, initial_density, final_density,
                            initial_percentile=0,
                            final_percentile=fp
                        )

                        with h5py.File(ridge_file, "w") as f:
                            f.create_dataset("ridges", data=ridges_cut)
                            f.create_dataset("initial_density", data=initial_density)
                            f.create_dataset("final_density", data=final_density)

                        if density_map is not None:
                            results_plot(density_map, ridges_cut, plot_file)

                        summary["completed_fp_outputs"].append({"run": run_id, "fp": fp})
                        print(f"[rank 0] Saved ridges → {ridge_file}")
                        print(f"[rank 0] Saved plot  → {plot_file}")

                    except Exception as e:
                        summary["errors_fp_outputs"].append({"run": run_id, "fp": fp, "error": str(e)})

                COMM.barrier()

                # ==================================================
                # Stage 2: contraction for each fp output
                # ==================================================
                contracted_file = ridge_file.replace(".h5", "_contracted.h5")

                # Skip contraction if exists
                con_exists = True
                if RANK == 0:
                    con_exists = os.path.exists(contracted_file)
                    if con_exists:
                        summary["skipped_existing_contraction"].append({"run": run_id, "fp": fp})
                con_exists = COMM.bcast(con_exists, root=0)
                if con_exists:
                    continue

                # Need ridge_file to exist
                ridge_ok = True
                if RANK == 0:
                    ridge_ok = os.path.exists(ridge_file)
                ridge_ok = COMM.bcast(ridge_ok, root=0)
                if not ridge_ok:
                    continue

                # Run contraction on rank 0 
                if RANK == 0:
                    try:
                        _ = process_ridge_file_local(
                            ridge_file,
                            mask,
                            nside,
                            radius_arcmin,
                            min_coverage,
                            ridges_dir_fp,
                            plots_dir
                        )
                        summary["completed_contraction"].append({"run": run_id, "fp": fp})
                    except Exception as e:
                        summary["errors_contraction"].append({"run": run_id, "fp": fp, "error": str(e)})

                COMM.barrier()

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------
    if RANK == 0:
        summary_file = os.path.join(output_base, "summary.txt")

        with open(summary_file, "w") as f:

            def block(title, arr):
                f.write(f"\n{title}:\n")
                print(f"\n{title}:")
                for x in arr:
                    f.write(str(x) + "\n")
                    print(" ", x)

            print("\n========== SUMMARY ==========")

            block("Raw ridges computed (find_filaments ran)", summary["completed_raw"])
            block("Raw ridges loaded from cache", summary["loaded_raw_cache"])
            block("Raw-stage errors", summary["errors_raw"])

            block("Completed fp outputs (ridges+plots)", summary["completed_fp_outputs"])
            block("Skipped existing fp outputs", summary["skipped_existing_fp_outputs"])
            block("FP-stage errors", summary["errors_fp_outputs"])

            block("Completed contraction outputs", summary["completed_contraction"])
            block("Skipped existing contraction outputs", summary["skipped_existing_contraction"])
            block("Contraction errors", summary["errors_contraction"])

            print("\nSummary saved to:", summary_file)
            f.write("\n===============================\n")

        print("================================\n")


if __name__ == "__main__":
    main()
