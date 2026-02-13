
import numpy as np
import h5py
import treecorr
import matplotlib.pyplot as plt
import pandas as pd



plt.rcParams.update({
    "figure.figsize": (8, 6.8),
    "figure.dpi": 100,

    "axes.linewidth": 1.6,
    "axes.labelsize": 15,
    "axes.titlesize": 15,

    # Major ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,

    # Minor ticks
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,

    "font.family": "serif",

    # "legend.frameon": False,
    "legend.fontsize": 12,

    "savefig.bbox": "tight",
})



ridge_shear_file = "../cosmo_sims/Cosmo_sim_ridges/S8/run_4/band_0.1/Ridges_final_p15/shear/shear_p15.csv"
ridge_shear = pd.read_csv(ridge_shear_file)


lens_filename = "../lhc_cosmo_sims_zero_err/S8_perp/run_4/lens_catalog_0.npy"
source_filename = "../lhc_cosmo_sims_zero_err/S8_perp/run_4/source_catalog_cutzl04.h5"

lens_file = h5py.File(lens_filename, "r")
source_file = h5py.File(source_filename, "r")


lens_ra = lens_file['RA'][:]
lens_dec = lens_file['DEC'][:]
lens_z = lens_file['Z_TRUE'][:]
cut = lens_z < 0.4
lens_ra = lens_ra[cut]
lens_dec = lens_dec[cut]
lens_z = lens_z[cut]



source_ra = source_file['RA'][:]
source_dec = source_file['DEC'][:]
source_z = source_file['Z_TRUE'][:]
g1 = source_file['G1'][:]
g2 = source_file['G2'][:]
cut = source_z > 0.4
source_ra = source_ra[cut]
source_dec = source_dec[cut]
source_z = source_z[cut]
g1 = g1[cut]
g2 = g2[cut]


config = {
    "min_sep": 1.0,
    "max_sep": 60.0,
    "sep_units": "arcmin",
    "nbins": 20,
    "bin_slop": 0.01,
    "num_threads": 8,
    "verbose": 2,

}

lens_cat = treecorr.Catalog(ra=lens_ra[::10], dec=lens_dec[::10], ra_units="degrees", dec_units="degrees")
source_cat = treecorr.Catalog(ra=source_ra[::10], dec=source_dec[::10], g1=g1[::10], g2=-g2[::10], ra_units="degrees", dec_units="degrees")


ng = treecorr.NGCorrelation(config)
ng.process(lens_cat, source_cat, num_threads=config["num_threads"])


gamma_t, gamma_x, sigma = ng.calculateXi()


plt.figure(figsize=(5,5))
plt.semilogx(ng.meanr, gamma_t/1e-3, ':', lw=3, label="Galaxy-Galaxy\nLensing")
plt.semilogx(60*np.degrees(ridge_shear["Weighted_Real_Distance"]), ridge_shear["Weighted_g_plus"]/1e-3, lw=3, label="Tangential\nRidge Lensing")
plt.semilogx(60*np.degrees(ridge_shear["Weighted_Real_Distance"]), ridge_shear["Weighted_g_cross"]/1e-3, linestyle='--', lw=3, label="Cross\nRidge Lensing")
plt.axhline(0, ls="-", color="k")
plt.xlabel(r"$\theta$ (arcmin)")
plt.ylabel(r"$\gamma_t$  / $10^{-3}$")
plt.xlim(1, 50)
plt.legend(loc='lower right')
plt.savefig("ggl_comparison.pdf")

