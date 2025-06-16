import os
import numpy as np
import h5py
import dredge_mod
import healpy
import matplotlib.pyplot as plt

base_sim_dir = "lhc_run_sims"

def get_filaments(run_id, initial_min_percentage, bandwidth_str, neighbours):
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
    
    ridges = dredge_mod.filaments(coordinates,
                         bandwidth=float(bandwidth_str),
                         convergence=1e-5,
                         initial_min_percentage=initial_min_percentage,
                         distance='haversine',
                         n_neighbors=neighbours,
                         n_process=4,
                         plot_dir = f'plots_{initial_min_percentage}_{bandwidth_str}_{neighbours}',
                         mesh_size=None)


    return ridges

def results_plot(density_map, ridges_filename, plot_filename):
    healpy.cartview(density_map, min=0, lonra=[20, 50], latra=[-40, -10],)
    healpy.graticule()
    ridges = np.load(ridges_filename)
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



# ra.min(), ra.max()
# map_plots2(1)
# initial min percentages 10, 20, 30, 40, 50
# bandwidths 0.1, 0.25, 0.5, 1.0, 2.0
# n_neighbours 100, 250, 500, 1000, 2000
import sys

if __name__ == "__main__":

    i1 = int(sys.argv[1])
    i2 = int(sys.argv[2])
    i3 = int(sys.argv[3])

    percentages = [10, 20, 30, 40, 50]
    bandwidths = ["0.1", "0.25", "0.5", "1.0", "2.0"]
    neighbours = [100, 250, 500, 1000, 2000]
    initial_min_percentage = percentages[i1]
    bandwidth_str = bandwidths[i2]
    neighbours = neighbours[i3]


    ridges = get_filaments(1, initial_min_percentage, bandwidth_str, neighbours)
    filename = f"ridges_run_{initial_min_percentage}_{bandwidth_str}_{neighbours}.npy", ridges
    np.save(filename, ridges)

    run_dir = os.path.join(base_sim_dir, "run_{1}")
    cat_file = os.path.join(run_dir, "lens_catalog_0.npy")
    density_map = build_density_map(cat_file, 2048, 0.5)

    plot_filename = f"filaments_run_{initial_min_percentage}_{bandwidth_str}_{neighbours}.png"

    results_plot(density_map, filename, plot_filename)


