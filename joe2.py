import os
import numpy as np
import h5py
import dredge_mod
import healpy
import matplotlib.pyplot as plt
import sys

base_sim_dir = "lhc_run_sims"

def get_filaments(run_id, bandwidth, neighbours):
    run_dir = os.path.join(base_sim_dir, f"run_{run_id}")
    source_catalog_file = os.path.join(run_dir, "lens_catalog_0.npy")

    print(f"\n--- Getting filaments for Run {run_id} in directory: {run_dir} ---")

    with h5py.File(source_catalog_file, 'r') as f:
        ra = f["RA"][:]
        dec = f["DEC"][:]
        z = f["Z_TRUE"][:]
    ra = (ra + 180) % 360
    
    coordinates = np.column_stack((dec, ra))
    print("Data size", coordinates.shape)
    
    ridges, initial_density, final_density = dredge_mod.filaments(coordinates,
                         bandwidth=bandwidth,
                         convergence=1e-5,
                         distance='haversine',
                         n_neighbors=neighbours,
                         n_process=4,
                         plot_dir = f'joe2/{neighbours}',
                         mesh_size=None)


    return ridges, initial_density, final_density

def results_plot(density_map, ridges, plot_filename):
    healpy.cartview(density_map, min=0, lonra=[20, 50], latra=[-40, -10],)
    healpy.graticule()
    ridges_ra = ridges[:, 1] - 180
    ridges_dec = ridges[:, 0]
    healpy.projplot(ridges_ra, ridges_dec, 'r.', markersize=1, lonlat=True)
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)


def build_density_map(filename, nside, smoothing_degrees=0.5):
    with h5py.File(filename, 'r') as f:
        ra = f["RA"][:]
        dec = f["DEC"][:]
    npix = healpy.nside2npix(nside)
    pix = healpy.ang2pix(nside, ra, dec, lonlat=True)
    m = np.zeros(npix, dtype=int)
    np.add.at(m, pix, 1)
    m1 = healpy.smoothing(m, fwhm=np.radians(smoothing_degrees), verbose=False)
    return m1



import sys

if __name__ == "__main__":

    i1 = int(sys.argv[1])

    bandwidth_str = "0.5"
    bandwidth = float(bandwidth_str)
    neighbours = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000]
    neighbours = neighbours[i1]

    print("Running version", neighbours)
    
    plot_filename = f"joe2/filaments_run_{neighbours}.png"
    filename = f"joe2/ridges_run_{neighbours}.npz"

    if os.path.exists(plot_filename):
        print("File already exists - skipping", plot_filename)
        sys.exit(0)
    
    ridges, initial_density, final_density = get_filaments(1, bandwidth, neighbours)
    np.savez(filename, ridges=ridges, initial_density=initial_density, final_density=final_density)

    cut1 = initial_density > initial_density.max() * 0.4
    cut2 = final_density > final_density.max() * 0.05
    ridges = ridges[cut1 & cut2]

    run_dir = os.path.join(base_sim_dir, "run_1")
    cat_file = os.path.join(run_dir, "lens_catalog_0.npy")
    density_map = build_density_map(cat_file, 2048, 0.5)

    results_plot(density_map, ridges, plot_filename)


