import os
import h5py
import numpy as np
import healpy
import matplotlib.pyplot as plt
import dredge_scms
from mpi4py.MPI import COMM_WORLD

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
    

def build_density_map(base_sim_dir, run_id, nside, smoothing_degrees=0.5):
    """
    Make a density map from the coordinates, automatically handling degrees/radians.
    """
    filename = os.path.join(base_sim_dir, f"run_{run_id}", "lens_catalog_0.npy")
    with h5py.File(filename, 'r') as f:
        ra = f["RA"][:]
        dec = f["DEC"][:]
        z_true = f["Z_TRUE"][:]

    mask = z_true < 0.4
    ra = ra[mask]
    dec = dec[mask]

    # Detect if values are in radians (max < 2π)
    if np.max(np.abs(ra)) < 2 * np.pi:
        ra = np.degrees(ra)
    if np.max(np.abs(dec)) < np.pi / 2 + 0.1:  # allow for some numerical margin
        dec = np.degrees(dec)

    # Healpy expects lonlat=True (RA in [0,360], DEC in [-90,90])
    npix = healpy.nside2npix(nside)
    pix = healpy.ang2pix(nside, ra, dec, lonlat=True)

    m = np.zeros(npix, dtype=int)
    np.add.at(m, pix, 1)
    m_smooth = healpy.smoothing(m, fwhm=np.radians(smoothing_degrees), verbose=False)
    return m_smooth



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


#def inspect_ridges(base_dir, base_label, run_id, final_percentile=15):
#    """
#    Load existing ridges + densities, apply safe cuts, and plot.
#    """
#    ridges_dir = f"simulation_ridges/{base_label}/band_0.3/Ridges_final_p{final_percentile:02d}"
#    h5_file = os.path.join(ridges_dir, f"{base_label}_run_{run_id}_ridges_p{final_percentile:02d}.h5")

#    if not os.path.exists(h5_file):
#        raise FileNotFoundError(f"Cannot find file: {h5_file}")

#    with h5py.File(h5_file, 'r') as f:
#        ridges = f["ridges"][:]
#        initial_density = f["initial_density"][:]
#        final_density = f["final_density"][:]

#    # Check and apply cuts safely
#    ridges_cut = redo_cuts_safe(ridges, initial_density, final_density,
#                                initial_percentile=0, final_percentile=final_percentile)

#    # Rebuild density map
#    density_map = load_density_map(base_dir, run_id)

#    # Create output directory for plots
#    out_plot_dir = os.path.join("lhc_run_sims_zero_err_10/debug_plots", f"{base_label}_run_{run_id}")
#    os.makedirs(out_plot_dir, exist_ok=True)

#    # --- Plot all ridges ---
#    plot_all = os.path.join(out_plot_dir, f"all_ridges_run{run_id}.png")
#    results_plot(density_map, ridges, plot_all, title=f"All Ridges — {base_label} run_{run_id}")

#    # --- Plot cut ridges ---
#    plot_cut = os.path.join(out_plot_dir, f"cut_ridges_run{run_id}_p{final_percentile:02d}.png")
#    results_plot(density_map, ridges_cut, plot_cut,
#                 title=f"Cut Ridges (>{final_percentile}%) — {base_label} run_{run_id}")


#if __name__ == "__main__":
#    base_sim_dir = "lhc_run_sims_zero_err_10"
#    base_label = "zero_err"
#    run_id = 1  # pick any run you want to inspect

#    inspect_ridges(base_sim_dir, base_label, run_id, final_percentile=15)





# ========================================================================
# ---------- # Apply change : Safe redo_cuts() with truncation -----------
# ========================================================================
def redo_cuts(ridges, initial_density, final_density,
              initial_percentile=0, final_percentile=25):
    """
    Truncate to minimum common length, apply percentile cuts,
    return (ridges_cut, initial_density_truncated, final_density_truncated).
    """
    n_r, n_i, n_f = len(ridges), len(initial_density), len(final_density)
    n_min = min(n_r, n_i, n_f)

    if len({n_r, n_i, n_f}) != 1:
        print(f"[WARNING] Length mismatch detected: ridges={n_r}, init={n_i}, final={n_f}")
        print(f"[INFO] Truncating all three arrays to n_min={n_min} to align them.")

    ridges_tr = ridges[:n_min]
    init_tr = initial_density[:n_min]
    final_tr = final_density[:n_min]

    cut1 = init_tr > np.percentile(init_tr, initial_percentile)
    cut2 = final_tr > np.percentile(final_tr, final_percentile)
    mask = cut1 & cut2

    print(f"[DEBUG] cut1 count={np.sum(cut1)}, cut2 count={np.sum(cut2)}, combined count={np.sum(mask)}")

    ridges_cut = ridges_tr[mask]
    return ridges_cut, init_tr, final_tr


def apply_cuts():
    base_label = "zero_err"
    base_sim_dir = "lhc_run_sims_zero_err_10"
    run_id = 1
    bandwidths = [0.3, 0.1]
    final_percentiles = [15]   # So that we can add more percentiles if needed

    for bw in bandwidths:
        home_dir = f"simulation_ridges/{base_label}/band_{bw:.1f}"
        for fp in final_percentiles:
            out_dir = os.path.join(home_dir, f"Ridges_final_p{fp:02d}")
            h5_filename = os.path.join(out_dir, f"{base_label}_run_{run_id}_ridges_p{fp:02d}.h5")
            plot_dir = os.path.join(home_dir, "plots_by_final_percentile")
            os.makedirs(plot_dir, exist_ok=True)

            if not os.path.exists(h5_filename):
                print(f"[SKIP] Missing file: {h5_filename}  (skipping bandwidth {bw}, fp {fp})")
                continue

            print(f"\nProcessing: {h5_filename}")

            # 1) Load existing arrays
            with h5py.File(h5_filename, "r") as f:
                # Expecting these datasets; if different, this will raise KeyError
                ridges = f["ridges"][:]
                initial_density = f["initial_density"][:]
                final_density = f["final_density"][:]

            # 2) Apply safe redo_cuts (truncates internally)
            ridges_cut, init_tr, final_tr = redo_cuts(
                ridges, initial_density, final_density,
                initial_percentile=0,
                final_percentile=fp
            )

            # 3) Backup original file then overwrite with post-cut datasets -> to be removed later
            backup_path = h5_filename + ".bak"
            try:
                shutil.copy2(h5_filename, backup_path)
                print(f"Backed up original to: {backup_path}")
            except Exception as e:
                print(f"[WARN] Could not create backup: {e}  (continuing, will attempt to overwrite)")

            with h5py.File(h5_filename, "w") as f:
                f.create_dataset("ridges", data=ridges_cut)
                # Save the aligned / truncated densities (length = n_min)
                f.create_dataset("initial_density", data=init_tr)
                f.create_dataset("final_density", data=final_tr)
            print(f"Overwrote (saved) post-cut HDF5 -> {h5_filename} (ridges: {len(ridges_cut)})")

            # 4) Rebuild density map for plotting and save plot of the cut ridges
            density_map = build_density_map(base_sim_dir, run_id, nside=512, smoothing_degrees=0.5)
            plot_path = os.path.join(plot_dir, f"{base_label}_run_{run_id}_Ridges_plot_p{fp:02d}.png")
            results_plot(density_map, ridges_cut, plot_path)
            print(f"Finished bandwidth {bw}, fp {fp}.")


if __name__ == "__main__":
    apply_cuts()


