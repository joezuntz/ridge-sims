# Your main script

import os
import numpy as np
import h5py
import dredge_mod  

base_sim_dir = "lhc_run_sims"
num_runs = 10

def process_run(run_id):
    run_dir = os.path.join(base_sim_dir, f"run_{run_id}")
    source_catalog_file = os.path.join(run_dir, "lens_catalog_0.npy")
    output_file = os.path.join(run_dir, f"filaments_run_{run_id}.hdf5")

    print(f"\n--- Processing Run {run_id} in directory: {run_dir} ---")

    try:
        with h5py.File(source_catalog_file, 'r') as f:
            ra = f["RA"][:]
            dec = f["DEC"][:]
            coordinates = np.column_stack((dec, ra))

            ridges = dredge_mod.filaments(coordinates,
                                       neighbors=10,
                                       bandwidth=None,
                                       convergence=0.005,
                                       percentage=None,
                                       distance='haversine',
                                       n_process=0,
                                       mesh_size=None)

            bandwidth_used = dredge_mod.get_last_bandwidth() # Get the bandwidth

            print(f"  Filaments found for Run {run_id} with shape: {ridges.shape}")
            print(f"  Bandwidth chosen for Run {run_id}: {bandwidth_used}")

            with h5py.File(output_file, 'w') as hf:
                hf.create_dataset('ridges', data=ridges)
                hf.attrs['bandwidth'] = bandwidth_used
                hf.attrs['run_id'] = run_id

            print(f"  Filaments and bandwidth saved to: {output_file}")

    except FileNotFoundError:
        print(f"  Error: Source catalog file not found at: {source_catalog_file}")
    except Exception as e:
        print(f"  Error processing Run {run_id}: {e}")

if __name__ == "__main__":
    for run_id in range(1, num_runs + 1):
        process_run(run_id)

    print("\n--- Finished processing all runs ---")