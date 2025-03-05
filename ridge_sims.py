import numpy as np
import os
import camb
from cosmology import Cosmology
import glass.shells
import glass.ext.camb
import healpy
import multiprocessing
import pickle
import sys
import tqdm

def flush():
    sys.stdout.flush()
    sys.stderr.flush()

# basic parameters of the simulation
nside = 4096
lmax = 10_000

# from https://arxiv.org/pdf/2105.13546
maglim_bias = [
    1.40,
    1.60,
    1.82,
    1.70,
    1.91,
    1.73,
]
redmagic_bias = [
    1.74,
    1.82,
    1.92,
    2.15,
    2.32,
]

# treat this as fixed; should be okay-ish
sigma_e = 0.27

# Number density per bin for DES (Giulia)
source_number_densities = [
    1.475584985490327, 
    1.479383426887689,
    1.483671693529899,
    1.461247850098986,
]

combined_source_number_densities = sum(source_number_densities)


def fiducial_params():
    h = 0.7
    Omega_c = 0.25
    Omega_b = 0.05
    return h, Omega_c, Omega_b

def get_parameter_objects(h, Omega_c, Omega_b):
    pars = camb.set_params(H0=100*h, omch2=Omega_c*h**2, ombh2=Omega_b*h**2,
                        NonLinear=camb.model.NonLinear_both)

    # get the cosmology from CAMB
    cosmo = Cosmology.from_camb(pars)

    return pars, cosmo

def fiducial_cosmology_objetcts():
    h, Omega_c, Omega_b = fiducial_params()
    pars, cosmo = get_parameter_objects(h, Omega_c, Omega_b)
    return pars, cosmo



def generate_shell_cl(windows, h, Omega_c, Omega_b, filename):

    pars, cosmo = get_parameter_objects(h, Omega_c, Omega_b)
    # set up CAMB parameters for matter angular power spectrum
    # shells of 150 Mpc in comoving distance spacing
    print("Computing C_ell")
    # # compute the angular matter power spectra of the shells with CAMB
    c_ells = glass.ext.camb.matter_cls(pars, lmax, windows)

    np.save(filename, c_ells)



def load_mask(mask_file):
    input_mask_nside = 4096
    hit_pix = np.load(mask_file)
    mask = np.zeros(healpy.nside2npix(input_mask_nside))
    mask[hit_pix] = 1
    # degrade mask to nside quality of simulation
    mask = healpy.reorder(mask, n2r = True)
    mask = healpy.ud_grade(mask, nside_out = nside)
    return mask



def generate_matter_fields(shell_cl, rng, g_ell_file=None):

    if g_ell_file is None or not os.path.exists(g_ell_file):
        shell_cl = glass.discretized_cls(shell_cl, nside=nside, lmax=lmax, ncorr=3)
        print("Discretized cl")
        with multiprocessing.Pool(8) as pool:
            # compute Gaussian spectra for lognormal fields from discretised spectra
            gls = glass.lognormal_gls(shell_cl, pool=pool)
            if g_ell_file is not None:
                with open(g_ell_file, "wb") as f:
                    pickle.dump(gls, f)
    else:
        with open(g_ell_file, "rb") as f:
            gls = pickle.load(f)

    print("Generated lognormal c_ell")

    # generator for lognormal matter fields
    matter = glass.generate_lognormal(gls, nside, ncorr=3, rng=rng)
    print("Generated lognormal fields")

    return matter

def shell_configuration(cosmo):
    zgrid = glass.shells.distance_grid(cosmo, 0., 3., dx=150.)

    # linear window functions for shells
    windows = glass.shells.linear_windows(zgrid)
    return windows


def fiducial_sim_step1():
    """
    This is very slow.
    """
    shell_cl_filename = "cls_200M_l10000.npy"
    h, Omega_c, Omega_b = fiducial_params()
    windows = shell_configuration(h, Omega_c, Omega_b)
    generate_shell_cl(windows, h, Omega_c, Omega_b, shell_cl_filename)

def load_number_densities(lens_type, combined=True):
    if combined:
        source_data = np.loadtxt("des-data/combined_source_nz.txt").T
    else:
        source_data = np.loadtxt("des-data/source_nz.txt").T

    source_z = source_data[0]
    source_nz = source_data[1:]

    if lens_type == "maglim":
        lens_data = np.loadtxt("des-data/maglim_nz.txt").T
    elif lens_type == "redmagic":
        lens_data = np.loadtxt("des-data/redmagic_nz.txt").T
    else:
        raise ValueError("lens_type must be 'maglim' or 'redmagic'")
    lens_z = lens_data[0]
    lens_nz = lens_data[1:]
    return source_z, source_nz, lens_z, lens_nz

def generate_shell_source_sample(delta_i, kappa_map, g1_map, g2_map, window, shell_ngal, shell_bias, sigma_e, mask, rng):
    
    # compute the lensing maps for this shell
    dtype=[
        ("RA", float),
        ("DEC", float),
        ("Z_TRUE", float),
        ("PHZ", float),
        ("ZBIN", int),
        ("G1", float),
        ("G2", float),
    ]

    catalog = []
    
    # generate source galaxies
    for gal_lon, gal_lat, gal_count in glass.positions_from_delta(
        shell_ngal,
        delta_i,
        shell_bias, # this shouldn't actually matter for the source sample
        vis=mask,
        rng=rng,
    ):  
        print("made source chunk of size", gal_count)
        gal_z = glass.redshifts(gal_count, window, rng=rng)
        # generate galaxy ellipticities from the chosen distribution
        gal_eps = glass.ellipticity_intnorm(gal_count, sigma_e, rng=rng)
        # apply the shear fields to the ellipticities
        gal_she = glass.galaxy_shear(
            gal_lon,
            gal_lat,
            gal_eps,
            kappa_map,
            g1_map,
            g2_map,
        )
        # make a mini-catalogue for the new rows
        rows = np.empty(gal_count, dtype=dtype)
        rows["RA"] = gal_lon
        rows["DEC"] = gal_lat
        rows["Z_TRUE"] = gal_z
        rows["G1"] = gal_she.real
        rows["G2"] = gal_she.imag

        catalog.append(rows)

    if len(catalog) == 0:
        return np.zeros(0, dtype=dtype)

    return np.concatenate(catalog)



def generate_shell_lens_sample(delta_i, window, shell_ngal, shell_bias, mask, rng):
    
    # compute the lensing maps for this shell
    dtype=[
        ("RA", float),
        ("DEC", float),
        ("Z_TRUE", float),
        ("ZBIN", int),
    ]

    catalog = []
    
    # generate lens galaxies
    for gal_lon, gal_lat, gal_count in glass.positions_from_delta(
        shell_ngal,
        delta_i,
        shell_bias,
        vis=mask,
        rng=rng,
    ):
        print("made lens chunk of size", gal_count)
        gal_z = glass.redshifts(gal_count, window, rng=rng)
        # generate galaxy ellipticities from the chosen distribution

        rows = np.empty(gal_count, dtype=dtype)
        rows["RA"] = gal_lon
        rows["DEC"] = gal_lat
        rows["Z_TRUE"] = gal_z

        catalog.append(rows)

    if len(catalog) == 0:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(catalog)


def fiducial_sim_step2(lens_type="maglim", tomographic=True):
    shell_cl_filename = "sim-data/cls_150M_l10000.npy"
    mask_filename = "des-data/desy3_gold_mask.npy"
    g_ell_file = "sim-data/g_ell_150M_l10000.pkl"


    source_z, source_nz, lens_z, lens_nz = load_number_densities(lens_type, tomographic)

    if os.path.exists(g_ell_file):
        shell_cl = None
    else:
        shell_cl = np.load(shell_cl_filename)

    print(f"Loaded shell c_ells")
    mask = load_mask(mask_filename)
    print("Loaded mask")
    _, cosmo = fiducial_cosmology_objetcts()
    flush()

    rng = np.random.default_rng(seed=42)
    matter = generate_matter_fields(shell_cl, rng, g_ell_file=g_ell_file)
    print("Matter field generator ready")

    if lens_type == "maglim":
        lens_number_densities = [
            0.15,
            0.107,
            0.109,
            0.146,
            0.106,
            0.1,
        ]
        galaxy_bias = maglim_bias
    else:
        # redmagic
        lens_number_densities = [
            0.022,
            0.038,
            0.058,
            0.029,
            0.025,
        ]
        galaxy_bias = redmagic_bias

    flush()

    # this will compute the convergence field iteratively
    convergence = glass.MultiPlaneConvergence(cosmo)
    print("Convergence object ready")

    windows = shell_configuration(cosmo)
    print(f"Loaded number densities - {len(source_nz)} source bins, {len(lens_nz)} lens bins")

    # normalize the number densities

    nbin_source = len(source_nz)
    nbin_lens = len(lens_nz)

    for i in range(nbin_source):
        source_nz[i] /= np.trapezoid(source_nz[i], source_z)
        source_nz *= source_number_densities[i]
    
    for i in range(nbin_lens):
        lens_nz[i] /= np.trapezoid(lens_nz[i], lens_z)
        lens_nz *= lens_number_densities[i]

    # distribute the n(z) for this bin over the 
    source_ngal_per_shell = [glass.partition(source_z, source_nz[b], windows) for b in range(nbin_source)]
    lens_ngal_per_shell = [glass.partition(lens_z, lens_nz[b], windows) for b in range(nbin_lens)]
    print("Number densities ready")

    source_catalogs = [[] for _ in range(nbin_source)]
    lens_catalogs = [[] for _ in range(nbin_lens)]

    # simulate the matter fields in the main loop, and build up the catalogue
    for i, delta in tqdm.tqdm(enumerate(matter)):
        print(f"Processing shell {i} / {len(windows)}")
        flush()
        window = windows[i]

        convergence.add_window(delta, window)
        kappa = convergence.kappa
        g1, g2 = glass.shear_from_convergence(kappa)

        for j in range(nbin_source):
            # ignore bias in the source sample
            bias = 1.0
            ngal = source_ngal_per_shell[j][i]
            print(f"Shell {i}, bin {j}, source ngal = {ngal}")
            if ngal != 0:
                rows = generate_shell_source_sample(delta, kappa, g1, g2, window, ngal, bias, sigma_e, mask, rng)
                source_catalogs[j].append(rows)
        
        for j in range(nbin_lens):
            ngal = lens_ngal_per_shell[j][i]

            # constant bias across the redshift bin
            bias = galaxy_bias[j] 

            print(f"Shell {i}, bin {j}, lens ngal = {ngal}")
            if ngal != 0:
                rows = generate_shell_lens_sample(delta, window, ngal, bias, mask, rng)
                lens_catalogs[j].append(rows)
        

    for j in range(nbin_source):
        source_catalog = np.concatenate(source_catalogs[j])
        np.save(f"source_catalog_{j}.npy", source_catalog)
    
    for j in range(nbin_lens):
        lens_catalog = np.concatenate(lens_catalogs[j])
        np.save(f"lens_catalog_{lens_type}_{j}.npy", lens_catalog)


if __name__ == "__main__":
    fiducial_sim_step2()
