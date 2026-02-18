import os, sys
import numpy as np
import h5py
from mpi4py import MPI

# PATH SETUP
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
# FILAMENT PIPELINE : Re created because ridge analysis tool was in use but can be unified with the version there. 
def run_filament_pipeline_seeded(
    bandwidth,
    base_sim_dir,
    run_ids,
    base_label,
    home_dir,
    N=2,
    z_cut=0.4,
    fraction=None,
    seed=3482364,
    neighbours=5000,
    convergence=1e-5,
    resume=True,
):
    mesh_size = int(N * 5e5)

    os.makedirs(home_dir, exist_ok=True)

    # isolate checkpoints by repeat and seed
    checkpoint_dir = os.path.join(home_dir, "checkpoints", f"seed_{seed}")
    if RANK == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    COMM.barrier()

    for run_id in run_ids:
        coordinates = None
        if RANK == 0:
            coordinates = load_coordinates(base_sim_dir, run_id, z_cut=z_cut, fraction=fraction)
        coordinates = COMM.bcast(coordinates, root=0)

        ridges, initial_density, final_density = dredge_scms.find_filaments(
            coordinates,
            bandwidth=np.radians(bandwidth),
            convergence=np.radians(convergence),
            distance_metric="haversine",
            n_neighbors=neighbours,
            comm=COMM,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            seed=seed,
            mesh_size=mesh_size,
        )

        COMM.barrier()

        if RANK == 0:
            final_percentiles = [15]
            initial_percentile = 0

            density_map = build_density_map(base_sim_dir, run_id, 512, z_cut=z_cut)

            plot_dir = os.path.join(home_dir, "plots_by_final_percentile")
            os.makedirs(plot_dir, exist_ok=True)

            for fp in final_percentiles:
                ridges_cut = redo_cuts(
                    ridges, initial_density, final_density,
                    initial_percentile=initial_percentile,
                    final_percentile=fp,
                )

                out_dir = os.path.join(home_dir, f"Ridges_final_p{fp:02d}")
                os.makedirs(out_dir, exist_ok=True)

                h5_filename = os.path.join(out_dir, f"{base_label}_run_{run_id}_ridges_p{fp:02d}.h5")
                with h5py.File(h5_filename, "w") as f:
                    f.create_dataset("ridges", data=ridges_cut)
                    f.create_dataset("initial_density", data=initial_density)
                    f.create_dataset("final_density", data=final_density)

                plot_path = os.path.join(plot_dir, f"{base_label}_run_{run_id}_Ridges_plot_p{fp:02d}.png")
                results_plot(density_map, ridges_cut, plot_path)

        COMM.barrier()


# ==============================================================
# POOLING
# ==============================================================
def _read_dataset_for_pooling(h5_path):
   
    with h5py.File(h5_path, "r") as f:
        if "ridges" in f:
            return "ridges", f["ridges"][...]
        for k in f.keys():
            if isinstance(f[k], h5py.Dataset):
                return k, f[k][...]
    raise RuntimeError(f"No root dataset found in {h5_path}")


def pool_contracted_files(contracted_files, out_path):
    if not contracted_files:
        raise RuntimeError("No contracted files to pool.")

    ds0, a0 = _read_dataset_for_pooling(contracted_files[0])
    arrays = [a0]
    src = [np.full(a0.shape[0], 0, dtype=np.int32)]

    for i, fp in enumerate(contracted_files[1:], start=1):
        ds, a = _read_dataset_for_pooling(fp)
        if ds != ds0:
            raise RuntimeError(f"Dataset mismatch: '{ds0}' vs '{ds}' in {fp}")
        arrays.append(a)
        src.append(np.full(a.shape[0], i, dtype=np.int32))

    pooled = np.concatenate(arrays, axis=0)
    src = np.concatenate(src, axis=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset(ds0, data=pooled)
        f.create_dataset("source_file_index", data=src)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("source_files", data=np.array(contracted_files, dtype=dt))

    return pooled.shape[0]


# ==============================================================
# MAIN
# ==============================================================
def main():
    # Parameters
    N_list     = [7]            
    run_ids    = range(1, 2)
    bandwidth  = 0.1
    z_cut      = 0.4
    fraction   = None

    n_repeats  = 20
    base_seed  = 3482364

    resume = True  # set False for forced fresh runs

    sim_bases = {"zero_err": "lhc_run_sims_zero_err_10"}
    output_base = "parameter_test"

    # contraction
    radius_arcmin = 4.0
    min_coverage  = 0.9
    nside         = 512

    mask_filename = os.path.join(parent_dir, "des-data", "desy3_gold_mask.npy")
    mask = np.load(mask_filename) if RANK == 0 else None
    mask = COMM.bcast(mask, root=0)

    for base_label, base_folder in sim_bases.items():
        base_sim_dir = os.path.join(parent_dir, base_folder)

        for run_id in run_ids:
            for N in N_list:

                # folder structures
                mesh_root = os.path.join(
                    output_base,
                    f"run_{run_id}",
                    f"band_{bandwidth:.1f}",
                    f"mesh_{N}",
                )

                contracted_paths = []

                for repeat_id in range(n_repeats):
                    seed = base_seed + repeat_id

                    # repeat folders 
                    home_dir = os.path.join(mesh_root, f"repeat_{repeat_id:03d}")
                    ridges_dir = os.path.join(home_dir, "Ridges_final_p15")
                    plots_dir  = os.path.join(home_dir, "plots_by_final_percentile")

                    if RANK == 0:
                        os.makedirs(ridges_dir, exist_ok=True)
                        os.makedirs(plots_dir, exist_ok=True)

                    # 1) find ridges with this seed
                    run_filament_pipeline_seeded(
                        bandwidth=bandwidth,
                        base_sim_dir=base_sim_dir,
                        run_ids=[run_id],
                        base_label=base_label,
                        home_dir=home_dir,
                        N=N,
                        z_cut=z_cut,
                        fraction=fraction,
                        seed=seed,
                        resume=resume,
                    )

                    COMM.barrier()

                    # 2) contract ridges
                    ridge_file = os.path.join(
                        ridges_dir,
                        f"{base_label}_run_{run_id}_ridges_p15.h5"
                    )

                    if RANK == 0:
                        process_ridge_file_local(
                            ridge_file,
                            mask,
                            nside,
                            radius_arcmin,
                            min_coverage,
                            ridges_dir,
                            plots_dir,
                        )

                        # Contracted files path
                        contracted = os.path.join(
                            ridges_dir,
                            f"{base_label}_run_{run_id}_ridges_p15_contracted.h5"
                        )
                        if not os.path.exists(contracted):
                            print(f"[rank 0][WARN] Missing contracted file: {contracted}")
                        else:
                            contracted_paths.append(contracted)

                    COMM.barrier()

                # 3) pool 
                if RANK == 0:
                    pooled_dir = os.path.join(mesh_root, "pooled")
                    pooled_path = os.path.join(
                        pooled_dir,
                        f"{base_label}_run_{run_id}_ridges_p15_contracted_pooled.h5"
                    )

                    if contracted_paths:
                        nrows = pool_contracted_files(contracted_paths, pooled_path)
                        print(f"[rank 0] Wrote pooled contracted file → {pooled_path} (rows={nrows})")
                    else:
                        print("[rank 0][WARN] No contracted files collected; skipping pooling.")

                COMM.barrier()


if __name__ == "__main__":
    main()
