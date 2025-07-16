import os
import numpy as np
import h5py
import dredge_scms
import healpy
import matplotlib.pyplot as plt
from mpi4py.MPI import COMM_WORLD

def load_coordinates(base_sim_dir, run_id, shift=True):
    """
    Load coordinates from a catalog file.

    If shift is True, the right ascension (RA) is shifted by 180 degrees.
    Use shift=True when finding filaments, but not for building density maps.
    """
    filename = os.path.join(base_sim_dir, f"run_{run_id}", "lens_catalog_0.npy")
    with h5py.File(filename, 'r') as f:
        ra = f["RA"][:]
        dec = f["DEC"][:]

    if shift:
        ra = (ra + 180) % 360
    
    # Inputs must now be in radians!
    coordinates = np.column_stack((dec, ra))
    coordinates = np.radians(coordinates)
    return coordinates

base_sim_dir = "lhc_run_sims"


def results_plot(density_map, ridges, plot_filename):
    """
    Make a plot of a density map and ridge points on top.
    """
    healpy.cartview(density_map, min=0, lonra=[20, 50], latra=[-30, 0],)
    healpy.graticule()

    ridges = np.degrees(ridges)
    ridges_ra = ridges[:, 1] - 180
    ridges_dec = ridges[:, 0]
    healpy.projplot(ridges_ra, ridges_dec, 'r.', markersize=1, lonlat=True)
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)


def build_density_map(base_sim_dir, run_id, nside, smoothing_degrees=0.5):
    """
    Make a density maps from the coordinates.
    """
    # The healpy conventions are different and should not have
    # the 180 deg shift applied
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
    bandwidth = 0.5
    neighbours = 5000
    convergence = 1e-5
    seed = 3482364
    mesh_size = None
    run_id = 1
    checkpoint_dir = "example/checkpoints"
    filename = f"example/ridges_run_{neighbours}_{bandwidth}.npz"

    coordinates = load_coordinates(base_sim_dir, run_id)
        
    ridges, initial_density, final_density = dredge_scms.find_filaments(coordinates,
                            bandwidth=np.radians(bandwidth),
                            convergence=np.radians(convergence),
                            distance_metric='haversine',
                            n_neighbors=neighbours,
                            comm=COMM_WORLD,
                            checkpoint_dir=checkpoint_dir,
                            resume=True,
                            seed=seed,
                            mesh_size=mesh_size
                            )
    
    if COMM_WORLD.rank == 0:
        print(f"Ridges shape: {ridges.shape}")
        print(f"Initial density shape: {initial_density.shape}")
        print(f"Final density shape: {final_density.shape}")
        np.savez(filename, ridges=ridges, initial_density=initial_density, final_density=final_density)

    # You can make a useful plot like this:
    # density_map = build_density_map(base_sim_dir, 1, 2048, smoothing_degrees=0.5)
    # data = np.load("example/ridges_run_5000_0.5.npz")
    # ridges = data["ridges"]
    # results_plot(density_map, ridges, "example.png")

if __name__ == "__main__":
    main()
