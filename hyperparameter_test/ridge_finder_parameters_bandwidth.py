import os, sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpi4py import MPI

# ==============================================================
# PATH SETUP
# ==============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

os.chdir(current_dir)

import dredge_scms
from ridge_analysis_tools import *

COMM = MPI.COMM_WORLD
RANK = COMM.rank


# ==============================================================
# FUNCTION 
# ==============================================================
def process_ridge_file_local(
    ridge_file, mask, nside,
    radius_arcmin, min_coverage,
    output_dir, plot_dir
):
    with h5py.File(ridge_file, "r") as f:
        ridges = f["ridges"][:]

    ridge_dec = ridges[:, 0]
    ridge_ra  = ridges[:, 1]
    n_total   = len(ridges)

    keep_idx = ridge_edge_filter_disk(
        ridge_ra, ridge_dec, mask, nside,
        radius_arcmin=radius_arcmin,
        min_coverage=min_coverage
    )

    ridges_clean = ridges[keep_idx]
    print(f"[contracted] {os.path.basename(ridge_file)}: "
          f"kept {len(ridges_clean)}/{n_total}")

    base_name = os.path.basename(ridge_file).replace(".h5", "_contracted.h5")
    out_file  = os.path.join(output_dir, base_name)

    with h5py.File(out_file, "w") as f:
        f.create_dataset("ridges", data=ridges_clean)

    plot_file = os.path.join(
        plot_dir,
        os.path.basename(out_file).replace(".h5", "_diagnostic.png")
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(ridge_ra, ridge_dec, s=1, alpha=0.3, label="All ridges")
    plt.scatter(
        ridges_clean[:, 1], ridges_clean[:, 0],
        s=1, alpha=0.6, label="Filtered ridges"
    )
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title(
        f"Contracted ridges\n"
        f"radius={radius_arcmin} arcmin, min_cov={min_coverage}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    plt.close()

    print(f"[plot] Saved diagnostic → {plot_file}")


# ==============================================================
# MAIN
# ==============================================================
def main():

    # ----------------------------------------------------------
    # PARAMETERS
    # ----------------------------------------------------------
    N_fixed = 2
    bandwidth_list = [0.01, 0.05, 0.1] #np.linspace(0.15, 0.55, 5)
    run_ids = range(1, 2)

    sim_bases = {
        "zero_err": "lhc_run_sims_zero_err_10",
    }

    output_base = "parameter_test"
    if RANK == 0:
        os.makedirs(output_base, exist_ok=True)

    # ----------------------------------------------------------
    # CONTRACTION PARAMETERS
    # ----------------------------------------------------------
    radius_arcmin = 4.0
    min_coverage  = 0.9
    nside         = 512

    mask_filename = os.path.join(
        parent_dir, "des-data", "desy3_gold_mask.npy"
    )
    mask = np.load(mask_filename) if RANK == 0 else None
    mask = COMM.bcast(mask, root=0)

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    summary = {
        "completed": [],
        "skipped_existing_output": [],
        "skipped_missing_input": [],
        "errors": [],

        "completed_contraction": [],
        "skipped_existing_contracted_output": [],
        "skipped_missing_contraction_input": [],
        "errors_contraction": []
    }

    # ----------------------------------------------------------
    # MAIN LOOPS
    # ----------------------------------------------------------
    for base_label, base_folder in sim_bases.items():

        base_sim_dir = os.path.join(parent_dir, base_folder)

        for run_id in run_ids:

            for bandwidth in bandwidth_list:

                N = N_fixed

                # --------------------------------------------------
                # DIRECTORY STRUCTURE
                # parameter_test/run_X/mesh_2/band_Y/
                # --------------------------------------------------
                home_dir = os.path.join(
                    output_base,
                    f"run_{run_id}",
                    f"mesh_{N}",
                    f"band_{bandwidth:.2f}"
                )

                ridges_dir = os.path.join(home_dir, "Ridges_final_p15")
                plots_dir  = os.path.join(home_dir, "plots_by_final_percentile")

                if RANK == 0:
                    os.makedirs(ridges_dir, exist_ok=True)
                    os.makedirs(plots_dir, exist_ok=True)

                ridge_file = os.path.join(
                    ridges_dir,
                    f"{base_label}_run_{run_id}_ridges_p15.h5"
                )

                # --------------------------------------------------
                # SAFETY CHECK — PIPELINE EXISTS
                # --------------------------------------------------
                exists = os.path.exists(ridge_file) if RANK == 0 else None
                exists = COMM.bcast(exists, root=0)

                pipeline_should_run = not exists

                if exists and RANK == 0:
                    summary["skipped_existing_output"].append(
                        {"run": run_id, "mesh": N, "bandwidth": float(bandwidth)}
                    )

                # --------------------------------------------------
                # INPUT CHECK
                # --------------------------------------------------
                if pipeline_should_run:
                    input_ok = True
                    if RANK == 0:
                        try:
                            _ = load_coordinates(base_sim_dir, run_id)
                        except FileNotFoundError:
                            input_ok = False
                            summary["skipped_missing_input"].append(
                                {"run": run_id, "mesh": N, "bandwidth": float(bandwidth)}
                            )
                    input_ok = COMM.bcast(input_ok, root=0)
                    pipeline_should_run = input_ok

                # --------------------------------------------------
                # STAGE 1 — FILAMENT PIPELINE
                # --------------------------------------------------
                if pipeline_should_run:
                    try:
                        run_filament_pipeline(
                            bandwidth=bandwidth,
                            base_sim_dir=base_sim_dir,
                            run_ids=[run_id],
                            base_label=base_label,
                            home_dir=home_dir,
                            N=N
                        )

                        if RANK == 0:
                            summary["completed"].append(
                                {"run": run_id, "mesh": N, "bandwidth": float(bandwidth)}
                            )

                    except Exception as e:
                        if RANK == 0:
                            summary["errors"].append(
                                {"run": run_id, "mesh": N,
                                 "bandwidth": float(bandwidth),
                                 "error": str(e)}
                            )
                        COMM.barrier()
                        continue

                COMM.barrier()

                # ==================================================
                # STAGE 2 — CONTRACTION
                # ==================================================
                contracted_file = ridge_file.replace(".h5", "_contracted.h5")

                exists2 = os.path.exists(contracted_file) if RANK == 0 else None
                exists2 = COMM.bcast(exists2, root=0)

                if exists2:
                    if RANK == 0:
                        summary["skipped_existing_contracted_output"].append(
                            {"run": run_id, "mesh": N, "bandwidth": float(bandwidth)}
                        )
                    continue

                ridge_ok = os.path.exists(ridge_file) if RANK == 0 else None
                ridge_ok = COMM.bcast(ridge_ok, root=0)

                if not ridge_ok:
                    if RANK == 0:
                        summary["skipped_missing_contraction_input"].append(
                            {"run": run_id, "mesh": N, "bandwidth": float(bandwidth)}
                        )
                    continue

                try:
                    if RANK == 0:
                        process_ridge_file_local(
                            ridge_file,
                            mask,
                            nside,
                            radius_arcmin,
                            min_coverage,
                            ridges_dir,
                            plots_dir
                        )

                        summary["completed_contraction"].append(
                            {"run": run_id, "mesh": N, "bandwidth": float(bandwidth)}
                        )

                except Exception as e:
                    if RANK == 0:
                        summary["errors_contraction"].append(
                            {"run": run_id, "mesh": N,
                             "bandwidth": float(bandwidth),
                             "error": str(e)}
                        )

    # ==========================================================
    # FINAL SUMMARY
    # ==========================================================
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

            block("Completed pipeline runs", summary["completed"])
            block("Skipped (existing pipeline output)", summary["skipped_existing_output"])
            block("Skipped (missing pipeline input)", summary["skipped_missing_input"])
            block("Pipeline errors", summary["errors"])

            block("Completed contraction runs", summary["completed_contraction"])
            block("Skipped (existing contracted output)", summary["skipped_existing_contracted_output"])
            block("Skipped (missing contraction input)", summary["skipped_missing_contraction_input"])
            block("Contraction errors", summary["errors_contraction"])

            f.write("\n===============================\n")

        print("\nSummary saved to:", summary_file)
        print("================================\n")


# ==============================================================
# ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    main()
