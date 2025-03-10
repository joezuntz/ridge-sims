import numpy as np
import camb
from cosmology import Cosmology
import glass.shells
import glass.ext.camb
import pickle
import tqdm
from .tools import flush

def get_parameter_objects(h, Omega_c, Omega_b):
    # get a CAMB parameter object and a glass Cosmology object
    pars = camb.set_params(H0=100*h, omch2=Omega_c*h**2, ombh2=Omega_b*h**2,
                        NonLinear=camb.model.NonLinear_both)
    cosmo = Cosmology.from_camb(pars)
    return pars, cosmo



def generate_shell_cl(windows, h, Omega_c, Omega_b, filename, lmax):
    # set up CAMB parameters for matter angular power spectrum
    pars, _ = get_parameter_objects(h, Omega_c, Omega_b)
    c_ells = glass.ext.camb.matter_cls(pars, lmax, windows)
    np.save(filename, c_ells)
    return c_ells


def generate_lognormal_gls(shell_cl, g_ell_file, nside, lmax, pool=None):
    shell_cl = glass.discretized_cls(shell_cl, nside=nside, lmax=lmax, ncorr=3)

    shift = 1.0
    n = glass.nfields_from_nspectra(len(shell_cl))
    fields = [glass.grf.Lognormal(shift) for _ in range(n)]
    gls = glass.solve_gaussian_spectra(fields, shell_cl, pool=pool)

    gls = glass.lognormal_gls(shell_cl)
    with open(g_ell_file, "wb") as f:
        pickle.dump(gls, f)
    
    return gls


def generate_matter_fields(gls, rng, nside):
    # generator for lognormal matter fields
    matter = glass.generate_lognormal(gls, nside, ncorr=3, rng=rng)
    return matter

def shell_configuration(cosmo, zmax, dx):
    zgrid = glass.shells.distance_grid(cosmo, 0., zmax, dx=dx)
    windows = glass.shells.linear_windows(zgrid)
    return windows


def generate_shell_source_sample(delta_i, kappa_map, g1_map, g2_map, window, shell_ngal, shell_bias, sigma_e, mask, rng):
    
    # compute the lensing maps for this shell
    dtype=[
        ("RA", float),
        ("DEC", float),
        ("Z_TRUE", float),
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






def simulate_catalogs(gls, rng, cosmo, sample, mask, nside, source_cat_file, lens_cat_file, zmax, dx):
    # Generate the iterator that produces the matter fields
    matter = generate_matter_fields(gls, rng, nside)

    # this will compute the convergence field iteratively
    convergence = glass.MultiPlaneConvergence(cosmo)
    windows = shell_configuration(cosmo, zmax, dx)

    # distribute the n(z) for this bin over the
    source_ngal_per_shell = [
        glass.partition(sample.source_z, sample.source_nz[b], windows)
        for b in range(sample.nbin_source)
    ]
    lens_ngal_per_shell = [
        glass.partition(sample.lens_z, sample.lens_nz[b], windows)
        for b in range(sample.nbin_lens)
    ]

    source_catalogs = [[] for _ in range(sample.nbin_source)]
    lens_catalogs = [[] for _ in range(sample.nbin_lens)]

    # simulate the matter fields in the main loop, and build up the catalogue
    for i, delta in tqdm.tqdm(enumerate(matter)):
        print(f"Processing shell {i} / {len(windows)}")
        flush()
        window = windows[i]

        convergence.add_window(delta, window)
        kappa = convergence.kappa
        g1, g2 = glass.shear_from_convergence(kappa)

        for j in range(sample.nbin_source):
            # ignore bias in the source sample
            bias = 1.0
            ngal = source_ngal_per_shell[j][i]
            print(f"Shell {i}, bin {j}, source ngal = {ngal}")
            if ngal != 0:
                rows = generate_shell_source_sample(
                    delta, kappa, g1, g2, window, ngal, bias, sample.sigma_e[j], mask, rng
                )
                source_catalogs[j].append(rows)

        for j in range(sample.nbin_lens):
            ngal = lens_ngal_per_shell[j][i]

            # constant bias across the redshift bin
            bias = sample.galaxy_bias[j]

            print(f"Shell {i}, bin {j}, lens ngal = {ngal}")
            if ngal != 0:
                rows = generate_shell_lens_sample(delta, window, ngal, bias, mask, rng)
                lens_catalogs[j].append(rows)

    for j in range(sample.nbin_source):
        source_catalog = np.concatenate(source_catalogs[j])
        np.save(source_cat_file.format(j), source_catalog)

    for j in range(sample.nbin_lens):
        lens_catalog = np.concatenate(lens_catalogs[j])
        np.save(
            lens_cat_file.format(j), lens_catalog
        )


