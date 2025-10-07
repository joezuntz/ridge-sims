import os
import h5py
import numpy as np
import healpy
import matplotlib.pyplot as plt


def results_plot(density_map, ridges, plot_filename):
    """
    Plot the density map and ridge points on top.
    """
    healpy.cartview(density_map, min=0, lonra=[20, 50], latra=[-30, 0])
    healpy.graticule()

    ridges_deg = np.degrees(ridges)
    ridges_ra = ridges_deg[:, 1] - 180
    ridges_dec = ridges_deg[:, 0]

    healpy.projplot(ridges_ra, ridges_dec, 'r.', markersize=1, lonlat=True)
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plot → {plot_filename}")


def redo_cuts_safe(ridges, initial_density, final_density,
                   initial_percentile=0, final_percentile=25):
    """
    Recompute the cuts safely, printing shape diagnostics if mismatch occurs.
    """
    n1, n2, n3 = len(ridges), len(initial_density), len(final_density)
    print(f"[DEBUG] Shapes: ridges={n1}, initial_density={n2}, final_density={n3}")

    # If mismatch, truncate to the smallest length
    n_min = min(n1, n2, n3)
    ridges = ridges[:n_min]
    initial_density = initial_density[:n_min]
    final_density = final_density[:n_min]

    cut1 = initial_density > np.percentile(initial_density, initial_percentile)
    cut2 = final_density > np.percentile(final_density, final_percentile)

    print(f"[DEBUG] Cut1 sum={np.sum(cut1)}, Cut2 sum={np.sum(cut2)}")
    ridges_cut = ridges[cut1 & cut2]
    print(f"[DEBUG] Resulting ridges_cut length={len(ridges_cut)}")

    return ridges_cut


def load_density_map(base_sim_dir, run_id, nside=512, smoothing_degrees=0.5):
    """
    Rebuild density map from the original catalog (for plotting background).
    """
    filename = os.path.join(base_sim_dir, f"run_{run_id}", "lens_catalog_0.npy")
    with h5py.File(filename, 'r') as f:
        ra = f["RA"][:]
        dec = f["DEC"][:]
        z_true = f["Z_TRUE"][:]
    mask = z_true < 0.4
    ra, dec = ra[mask], dec[mask]

    # Healpy expects lonlat in degrees
    npix = healpy.nside2npix(nside)
    pix = healpy.ang2pix(nside, ra, dec, lonlat=True)
    m = np.zeros(npix, dtype=int)
    np.add.at(m, pix, 1)
    m_smooth = healpy.smoothing(m, fwhm=np.radians(smoothing_degrees), verbose=False)
    return m_smooth


def inspect_ridges(base_dir, base_label, run_id, final_percentile=15):
    """
    Load existing ridges + densities, apply safe cuts, and plot.
    """
    ridges_dir = f"simulation_ridges/{base_label}/band_0.1/Ridges_final_p{final_percentile:02d}"
    h5_file = os.path.join(ridges_dir, f"{base_label}_run_{run_id}_ridges_p{final_percentile:02d}.h5")

    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"Cannot find file: {h5_file}")

    with h5py.File(h5_file, 'r') as f:
        ridges = f["ridges"][:]
        initial_density = f["initial_density"][:]
        final_density = f["final_density"][:]

    # Check and apply cuts safely
    ridges_cut = redo_cuts_safe(ridges, initial_density, final_density,
                                initial_percentile=0, final_percentile=final_percentile)

    # Rebuild density map
    density_map = load_density_map(base_dir, run_id)

    # Create output directory for plots
    out_plot_dir = os.path.join("lhc_run_sims_zero_err_10/debug_plots", f"{base_label}_run_{run_id}")
    os.makedirs(out_plot_dir, exist_ok=True)

    # --- Plot all ridges ---
    plot_all = os.path.join(out_plot_dir, f"all_ridges_run{run_id}.png")
    results_plot(density_map, ridges, plot_all, title=f"All Ridges — {base_label} run_{run_id}")

    # --- Plot cut ridges ---
    plot_cut = os.path.join(out_plot_dir, f"cut_ridges_run{run_id}_p{final_percentile:02d}.png")
    results_plot(density_map, ridges_cut, plot_cut,
                 title=f"Cut Ridges (>{final_percentile}%) — {base_label} run_{run_id}")


if __name__ == "__main__":
    base_sim_dir = "lhc_run_sims_zero_err_10"
    base_label = "zero_err"
    run_id = 1  # pick any run you want to inspect

    inspect_ridges(base_sim_dir, base_label, run_id, final_percentile=15)


if __name__ == "__main__":
    base_sim_dir = "lhc_run_sims_zero_err_10"
    base_label = "zero_err"
    run_id = 1  # pick any run you want to inspect

    inspect_ridges(base_sim_dir, base_label, run_id, final_percentile=15)
