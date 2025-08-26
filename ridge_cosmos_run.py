import os
import numpy as np
import h5py
import healpy
import matplotlib.pyplot as plt
from mpi4py.MPI import COMM_WORLD
import dredge_scms

# ======================================================
# Utilities
# ======================================================

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
    plt.close()


def build_density_map(base_sim_dir, run_id, nside, smoothing_degrees=0.5):
    """
    Make a density map from the coordinates.
    """
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

# ======================================================
# Main filament finding per cosmology
# ======================================================

def run_filaments(base_sim_dir, run_id, Omega_m, sigma8,
                  bandwidth=0.5, neighbours=5000, convergence=1e-5,
                  seed=3482364, mesh_size=int(5e5)):
    """
    Run filament finding for a given cosmology run_id and parameters.
    Saves results in a directory named with Omega_m and sigma8.
    """
    checkpoint_dir = f"example_cosmologies/checkpoints/run_{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

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
        # Percentiles to cut
        final_percentiles = [15]
        initial_percentile = 0

        density_map = build_density_map(base_sim_dir, run_id, 512)

        # Directory for this cosmology
        cosmo_tag = f"Om{Omega_m:.3f}_s8{sigma8:.3f}"
        base_out_dir = os.path.join("example_cosmologies", cosmo_tag)
        os.makedirs(base_out_dir, exist_ok=True)

        for fp in final_percentiles:
            ridges_cut = redo_cuts(ridges, initial_density, final_density,
                                   initial_percentile=initial_percentile,
                                   final_percentile=fp)

            # Save ridges
            out_dir = os.path.join(base_out_dir, f"Ridges_final_p{fp:02d}")
            os.makedirs(out_dir, exist_ok=True)

            h5_filename = os.path.join(out_dir, f"ridges_p{fp:02d}.h5")
            with h5py.File(h5_filename, 'w') as f:
                f.create_dataset("ridges", data=ridges_cut)
                f.create_dataset("initial_density", data=initial_density)
                f.create_dataset("final_density", data=final_density)
                f.attrs["Omega_m"] = Omega_m
                f.attrs["sigma8"] = sigma8
                f.attrs["run_id"] = run_id

            print(f"[rank 0] Saved ridges for final_percentile={fp} "
                  f"to {h5_filename} for Omega_m={Omega_m:.3f}, sigma8={sigma8:.3f}")

            # Plot
            plot_path = os.path.join(out_dir, f"Ridges_plot_p{fp:02d}.png")
            results_plot(density_map, ridges_cut, plot_path)

# ======================================================
# Loop over cosmologies
# ======================================================

def main():
    base_sim_dir = "cosmo_direct_runs"

    for i in range(7):
        sigma8 = 0.8 + (i - 5) * 0.01
        Omega_m = 0.3 + (i - 5) * 0.01
        run_id = i + 1

        if COMM_WORLD.rank == 0:
            print(f"Starting cosmology run {run_id} "
                  f"with Omega_m={Omega_m:.3f}, sigma8={sigma8:.3f}")

        run_filaments(base_sim_dir, run_id, Omega_m, sigma8)

        if COMM_WORLD.rank == 0:
            print(f"Finished cosmology run {run_id}")

    if COMM_WORLD.rank == 0:
        print("All cosmology runs completed.")


if __name__ == "__main__":
    main()
        