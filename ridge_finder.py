import os
import numpy as np
import h5py
import dredge_scms
import healpy
import matplotlib.pyplot as plt
from mpi4py.MPI import COMM_WORLD

import time

# Record the start time
start_time = time.perf_counter()

def load_coordinates(base_sim_dir, run_id, shift=True):
    filename = os.path.join(base_sim_dir, f"run_{run_id}", "lens_catalog_0.npy")
    with h5py.File(filename, 'r') as f:
        ra = f["RA"][:]
        dec = f["DEC"][:]
        z_true = f["Z_TRUE"][:]
    mask = z_true < 0.4
    ra = ra[mask]
    dec = dec[mask]
    if shift:
        ra = (ra + 180) % 360
    coordinates = np.radians(np.column_stack((dec, ra)))
    return coordinates

def build_density_map(base_sim_dir, run_id, nside, smoothing_degrees=0.5):
    data = load_coordinates(base_sim_dir, run_id, shift=False)
    dec = np.degrees(data[:, 0])
    ra = np.degrees(data[:, 1])
    npix = healpy.nside2npix(nside)
    pix = healpy.ang2pix(nside, ra, dec, lonlat=True)
    m = np.zeros(npix, dtype=int)
    np.add.at(m, pix, 1)
    m1 = healpy.smoothing(m, fwhm=np.radians(smoothing_degrees), verbose=False)
    return m1

def redo_cuts(ridges, initial_density, final_density, initial_percentile=0, final_percentile=25):
    cut1 = initial_density > np.percentile(initial_density, initial_percentile)
    cut2 = final_density > np.percentile(final_density, final_percentile)
    return ridges[cut1 & cut2]

def main():
    base_sim_dir = "lhc_run_sims_50"
    num_runs = 30
    final_percentile = 15
    initial_percentile = 0

    for run_id in range(1, num_runs + 1):
        print(f"--- Processing Run {run_id} ---")
        
        # Define directories for the current run
        output_base_dir = f"example30_band0.4/run_{run_id}"
        os.makedirs(output_base_dir, exist_ok=True)
        
        ridges_dir = os.path.join(output_base_dir, "ridges")
        os.makedirs(ridges_dir, exist_ok=True)
        
        checkpoint_dir = os.path.join(output_base_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load coordinates for the current run
        coordinates = load_coordinates(base_sim_dir, run_id)

        # Find filaments
        ridges, initial_density, final_density = dredge_scms.find_filaments(
            coordinates,
            bandwidth=np.radians(0.4),
            convergence=np.radians(1e-5),
            distance_metric='haversine',
            n_neighbors=5000,
            comm=COMM_WORLD,
            checkpoint_dir=checkpoint_dir,
            resume=True,
            seed=3482364,
            mesh_size=int(5e5)
        )

        if COMM_WORLD.rank == 0:
            ridges_cut = redo_cuts(ridges, initial_density, final_density,
                                   initial_percentile=initial_percentile,
                                   final_percentile=final_percentile)
            
            # Save ridges
            h5_filename = os.path.join(ridges_dir, f"ridges_p{final_percentile:02d}.h5")
            with h5py.File(h5_filename, 'w') as f:
                f.create_dataset("ridges", data=ridges_cut)
            print(f"Saved ridges for run {run_id} to {h5_filename}")

if __name__ == "__main__":
    main()
    
    
# Record the end time
end_time = time.perf_counter()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:0.4f} seconds")