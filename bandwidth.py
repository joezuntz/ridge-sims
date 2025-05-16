import os
import numpy as np
import h5py
import dredge_mod  

base_sim_dir = "lhc_run_sims"
num_runs = 10
all_bandwidths = {}

def get_bandwidth_for_run(run_id):
    run_dir = os.path.join(base_sim_dir, f"run_{run_id}")
    source_catalog_file = os.path.join(run_dir, "lens_catalog_0.npy")

    print(f"\n--- Getting Bandwidth for Run {run_id} in directory: {run_dir} ---")

    try:
        with h5py.File(source_catalog_file, 'r') as f:
            ra = f["RA"][:]
            dec = f["DEC"][:]
            coordinates = np.column_stack((dec, ra))

            dredge_mod.filaments(coordinates,
                               neighbors=10,
                               bandwidth=None,
                               convergence=0.005,
                               percentage=None,
                               distance='haversine',
                               n_process=0,
                               mesh_size=None)

            bandwidth_used = dredge_mod.get_last_bandwidth() # Get the bandwidth
            print(f"  Bandwidth chosen for Run {run_id}: {bandwidth_used}")
            return run_id, bandwidth_used

    except FileNotFoundError:
        print(f"  Error: Source catalog file not found at: {source_catalog_file}")
        return run_id, None
    except Exception as e:
        print(f"  Error processing Run {run_id}: {e}")
        return run_id, None

if __name__ == "__main__":
    for run_id in range(1, num_runs + 1):
        run_id, bandwidth = get_bandwidth_for_run(run_id)
        if bandwidth is not None:
            all_bandwidths[run_id] = bandwidth
        # We stop after recording the bandwidth for each run

    # Save all bandwidths to a single file in the base directory AFTER processing all runs
    bandwidths_output_file = os.path.join(base_sim_dir, "all_bandwidths.hdf5")
    with h5py.File(bandwidths_output_file, 'w') as hf_bandwidths:
        for run_id, bw in all_bandwidths.items():
            hf_bandwidths.create_dataset(f'run_{run_id}', data=np.array([bw]))

    print(f"\nAll bandwidths saved to: {bandwidths_output_file}")
    print("\n--- Finished getting bandwidths for all runs ---")