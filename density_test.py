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
        z_true = f["Z_TRUE"][:]
    mask = z_true < 0.4
    ra = ra[mask]
    dec = dec[mask]
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
    mesh_size = int(5e5)
    base_sim_dir = "lhc_run_sims"
    run_id = 1
    checkpoint_dir = "example_zl04_mesh5e5/checkpoints"
    
    coordinates = load_coordinates(base_sim_dir, run_id)

    ridges, initial_density, final_density = dredge_scms.find_filaments(
        coordinates,
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
        final_percentiles = [15] # [0, 10, 25, 40, 50, 60, 75, 85, 90, 95]
        initial_percentile = 0

        # Build density map once (outside the loop)
        density_map = build_density_map(base_sim_dir, run_id, 512)

        # directory for plots
        plot_dir = "example_zl04_mesh5e5/plots_by_final_percentile"
        os.makedirs(plot_dir, exist_ok=True)

        for fp in final_percentiles:
            ridges_cut = redo_cuts(ridges, initial_density, final_density,
                                   initial_percentile=initial_percentile,
                                   final_percentile=fp)

            # Save ridges
            out_dir = f"example_zl04_mesh5e5/Ridges_final_p{fp:02d}"
            os.makedirs(out_dir, exist_ok=True)

            h5_filename = os.path.join(out_dir, f"ridges_p{fp:02d}.h5")
            with h5py.File(h5_filename, 'w') as f:
                f.create_dataset("ridges", data=ridges_cut)
                f.create_dataset("initial_density", data=initial_density)
                f.create_dataset("final_density", data=final_density)

            print(f"[rank 0] Saved ridges for final_percentile={fp} to {h5_filename}")

            # plots
            plot_path = os.path.join(plot_dir, f"Ridges_plot_p{fp:02d}.png")
            results_plot(density_map, ridges_cut, plot_path)


if __name__ == "__main__":
    main()
