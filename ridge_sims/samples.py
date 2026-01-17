import numpy as np
import healpy
import yaml
from types import SimpleNamespace

class SampleInfo(SimpleNamespace):
    pass

mask_filename = "des-data/desy3_gold_mask.npy"
lsst_mask_filename_y1 = "lsst-data/lsst_y1_wfd_exgal_mask_nside_64.fits"
lsst_mask_filename_y10 = "lsst-data/lsst_y10_wfd_exgal_mask_nside_64.fits"

source_nz_filename = "des-data/source_nz_smooth.txt"
tomographic_maglim_nz_filename = "des-data/maglim_nz.txt"
tomographic_redmagic_nz_filename = "des-data/redmagic_nz.txt"

combined_maglim_nz_filename = "des-data/maglim_nz_zcut_0.9.txt"
combined_redmagic_nz_filename = "des-data/redmagic_nz_zcut_0.9.txt"


# from https://arxiv.org/pdf/2105.13546
tomographic_maglim_bias = [
    1.40,
    1.60,
    1.82,
    1.70,
    1.91,
    1.73,
]
tomographic_redmagic_bias = [
    1.74,
    1.82,
    1.92,
    2.15,
    2.32,
]

combined_maglim_bias = np.mean(tomographic_maglim_bias)
combined_redmagic_bias = np.mean(tomographic_redmagic_bias)

tomographic_maglim_number_densities = [
    0.15,
    0.107,
    0.109,
    0.146,
    0.106,
    0.1,
]

maglim_combined_number_densities = sum(tomographic_maglim_number_densities)

tomographic_redmagic_number_densities = [
    0.022,
    0.038,
    0.058,
    0.029,
    0.025,
]


redmagic_combined_number_densities = sum(tomographic_redmagic_number_densities)

tomographic_sigma_e = [0.27, 0.27, 0.27, 0.27]
combined_sigma_e = [np.mean(tomographic_sigma_e)]

# Number density per bin for DES (Giulia)
tomographic_source_number_densities = [
    1.475584985490327, 
    1.479383426887689,
    1.483671693529899,
    1.461247850098986,
]

combined_source_number_densities = sum(tomographic_source_number_densities)

def load_lsst_sample_information(lsst, combined):
    """
    Load LSST-like sample information from files from the LSST forecasting repository.

    Parameters
    ----------
    lsst : int
        If 1, use LSST Y1 number densities.
        If 10, use LSST Y10 number densities.
    combined : bool
        Whether to use the combined source and lens samples. Default is True.

    Returns
    -------
    sample : SampleInfo
        The object containing the number density information.
    
    """
    if lsst not in [1, 10]:
        raise ValueError("lsst must be 1 or 10 if not False/0")
    
    # Load the n_eff from the numpy files from forecasting. They are already in units of
    # objects per arcmin^2
    clustering_neff = np.load(f'lsst-data/lsst_n_eff_clustering_year_{lsst}.npy', allow_pickle=True).item()["n_gal"]
    lensing_neff = np.load(f'lsst-data/lsst_n_eff_lensing_year_{lsst}.npy', allow_pickle=True).item()["n_gal"]
    clustering_neff = np.array(clustering_neff)
    lensing_neff =  np.array(lensing_neff)
    print(clustering_neff.shape, lensing_neff.shape, clustering_neff.dtype, lensing_neff.dtype)

    nbin_source = len(lensing_neff)
    nbin_lens = len(clustering_neff)

    # Load the n(z) from the numpy files from forecasting. I think these are all normalized to 1
    # already, but just in case we will normalize them again below.
    source_nz_data = np.load(f"lsst-data/srd_source_bins_year_{lsst}.npy", allow_pickle=True).item()
    source_z = source_nz_data['redshift_range']
    source_nz = np.array([source_nz_data['bins'][i] for i in range(nbin_source)])

    lens_nz_data = np.load(f"lsst-data/srd_lens_bins_year_{lsst}.npy", allow_pickle=True).item()
    lens_z = lens_nz_data['redshift_range']
    lens_nz = np.array([lens_nz_data['bins'][i] for i in range(nbin_lens)])


    # Load galaxy bias from the yaml file from forecasting
    galaxy_bias = np.zeros(nbin_lens)
    with open(f"lsst-data/linear_galaxy_bias_parameters_y{lsst}.yaml") as f:
        galaxy_bias_data = yaml.safe_load(f)
        for i in range(nbin_lens):
            galaxy_bias[i] = galaxy_bias_data[f"b_{i+1}"]

    # In the forecasting they always use sigma_e = 0.26
    sigma_e  = np.full(lensing_neff.shape, 0.26)


    # Normalize both the source and lens
    for i in range(nbin_source):
        source_nz[i] /= np.trapezoid(source_nz[i], source_z)
        source_nz[i] *= lensing_neff[i]
    
    for i in range(nbin_lens):
        lens_nz[i] /= np.trapezoid(lens_nz[i], lens_z)
        lens_nz[i] *= clustering_neff[i]

    # If we are combining into a single tomographic bin then:
    if combined:
        # The n_effs just sum
        lensing_neff = [np.sum(lensing_neff)]
        clustering_neff = [np.sum(clustering_neff)]
        # The galaxy bias is the mean. We should ideally use
        # the weighted mean but this is close enough as there is
        # not much variation in the neffs
        galaxy_bias = [np.mean(galaxy_bias)]
        # This one is correct as the source sample has equal numbers
        # of objets in each bin by construction
        sigma_e = [np.mean(sigma_e)]

        # The n(z)'s sum because they are normalized by the neff above.
        source_nz = np.array([np.sum(source_nz, axis=0)])
        lens_nz = np.array([np.sum(lens_nz, axis=0)])
        # The counts are now just one bin each
        nbin_source = 1
        nbin_lens = 1

    # Collect everything into the SampleInfo object
    sample = SampleInfo()
    sample.nbin_source = nbin_source
    sample.nbin_lens = nbin_lens
    sample.source_z = source_z
    sample.source_nz = source_nz
    sample.lens_z = lens_z
    sample.lens_nz = lens_nz
    sample.lens_number_densities = lensing_neff
    sample.galaxy_bias = galaxy_bias
    sample.source_number_densities = clustering_neff
    sample.sigma_e = sigma_e

    return sample

    

def load_sample_information(lens_type, combined=True, lsst=False): 
    """
    Create and return a SampleInfo object with the number density information
    for the source and lens samples, and galaxy bias for the latter. 
    
    The number densities are taken from the DES Y3 data release.

    Parameters
    ----------
    lens_type : str
        The type of lens sample to use. Must be 'maglim' or 'redmagic'.

    combined : bool 
        Whether to use the combined source and lens samples. Default is True.
        If False use tomographic samples.

    lsst: int:
        If 0, use DES Y3 number densities.
        If 1, use LSST Y1 number densities.
        If 10, use LSST Y10 number densities.

    Returns
    -------
    sample : SampleInfo
        The object containing the number density information.
    """

    if lsst:
        return load_lsst_sample_information(lsst, combined)

    source_data = np.loadtxt(source_nz_filename).T

    source_z = source_data[0]
    source_nz = source_data[1:]

    if combined:
        source_number_densities = [combined_source_number_densities]
        sigma_e = combined_sigma_e
    else:
        source_number_densities = tomographic_source_number_densities
        sigma_e = tomographic_sigma_e


    if lens_type == "maglim":
        if combined:
            lens_number_densities = [maglim_combined_number_densities]
            galaxy_bias = [np.mean(combined_maglim_bias)]
            lens_data = np.loadtxt(combined_maglim_nz_filename).T
        else:
            sample.lens_number_densities = tomographic_maglim_number_densities
            sample.galaxy_bias = tomographic_maglim_bias
            lens_data = np.loadtxt(tomographic_maglim_nz_filename).T
            
    else:
        if combined:
            lens_number_densities = [redmagic_combined_number_densities]
            galaxy_bias = np.mean(tomographic_redmagic_bias)
            lens_data = np.loadtxt(combined_redmagic_nz_filename).T
        else:
            lens_number_densities = tomographic_redmagic_number_densities
            galaxy_bias = tomographic_redmagic_bias
            lens_data = np.loadtxt(tomographic_redmagic_nz_filename).T

    lens_z = lens_data[0]
    lens_nz = lens_data[1:]
    nbin_lens = len(lens_nz)
    nbin_source = len(source_nz)

    for i in range(nbin_source):
        source_nz[i] /= np.trapezoid(source_nz[i], source_z)
        source_nz[i] *= source_number_densities[i]
    
    for i in range(nbin_lens):
        lens_nz[i] /= np.trapezoid(lens_nz[i], lens_z)
        lens_nz[i] *= lens_number_densities[i]


    sample = SampleInfo()
    sample.nbin_source = nbin_source
    sample.nbin_lens = nbin_lens
    sample.source_z = source_z
    sample.source_nz = source_nz
    sample.lens_z = lens_z
    sample.lens_nz = lens_nz
    sample.lens_number_densities = lens_number_densities
    sample.galaxy_bias = galaxy_bias
    sample.source_number_densities = source_number_densities
    sample.sigma_e = sigma_e


    return sample



##################################################################################
#      A Test function to combine DES simulation and LSST Y10 nz samples
##################################################################################



def load_sample_information_advanced(lens_type, combined=True, lsst=False, lsst10_nz=False):  # NEW
    """
    Create and return a SampleInfo object with the number density information
    for the source and lens samples, and galaxy bias for the latter. 
    
    The number densities are taken from the DES Y3 data release.

    Parameters
    ----------
    lens_type : str
        The type of lens sample to use. Must be 'maglim' or 'redmagic'.

    combined : bool 
        Whether to use the combined source and lens samples. Default is True.
        If False use tomographic samples.

    lsst: int:
        If 0, use DES Y3 number densities.
        If 1, use LSST Y1 number densities.
        If 10, use LSST Y10 number densities.

    Returns
    -------
    sample : SampleInfo
        The object containing the number density information.
    """

    # Full LSST mode: switches *everything* (mask, n_eff, bias, sigma_e, etc.)
    if lsst:
        return load_lsst_sample_information(lsst, combined)

    # Hybrid mode: keep DES-like sample properties (densities, bias, sigma_e)
    # but replace the *shape* of n(z) with LSST Y10.
    # keep the DES mask and number densities but test the
    # sensitivity of ridge/shear measurements to a deeper redshift distribution.
    if lsst10_nz:  
        # Load LSST Y10 n(z) shapes (do NOT keep its n_eff normalization)
        y = 10
        source_nz_data = np.load(f"lsst-data/srd_source_bins_year_{y}.npy", allow_pickle=True).item()
        source_z = source_nz_data['redshift_range']
        source_nz = np.array(source_nz_data['bins'])

        lens_nz_data = np.load(f"lsst-data/srd_lens_bins_year_{y}.npy", allow_pickle=True).item()
        lens_z = lens_nz_data['redshift_range']
        lens_nz = np.array(lens_nz_data['bins'])

        # consistent 2D shapes: (nbin, nz)
        if source_nz.ndim == 1:
            source_nz = source_nz[None, :]
        if lens_nz.ndim == 1:
            lens_nz = lens_nz[None, :]

        # DES-like number densities and per-bin sigma_e
        if combined:
            source_number_densities = [combined_source_number_densities]
            sigma_e = combined_sigma_e
        else:
            source_number_densities = tomographic_source_number_densities
            sigma_e = tomographic_sigma_e

        if lens_type == "maglim":
            if combined:
                lens_number_densities = [maglim_combined_number_densities]
                galaxy_bias = [float(np.mean(combined_maglim_bias))]
            else:
                lens_number_densities = tomographic_maglim_number_densities
                galaxy_bias = tomographic_maglim_bias
        else:
            if combined:
                lens_number_densities = [redmagic_combined_number_densities]
                galaxy_bias = [float(np.mean(tomographic_redmagic_bias))]
            else:
                lens_number_densities = tomographic_redmagic_number_densities
                galaxy_bias = tomographic_redmagic_bias

        # Normalize LSST n(z) shapes to 1, then scale to DES densities.
        # Keep the shape of LSST Y10 but matches total counts to DES.
        nbin_source = source_nz.shape[0]
        nbin_lens = lens_nz.shape[0]

        # tomographic DES but LSST provides a different number
        # we take the first N bins. 
        
        if not combined:
            if len(source_number_densities) != nbin_source:
                n = min(len(source_number_densities), nbin_source)
                source_nz = source_nz[:n]
                source_number_densities = source_number_densities[:n]
                sigma_e = sigma_e[:n]
                nbin_source = n
            if len(lens_number_densities) != nbin_lens:
                n = min(len(lens_number_densities), nbin_lens)
                lens_nz = lens_nz[:n]
                lens_number_densities = lens_number_densities[:n]
                galaxy_bias = galaxy_bias[:n]
                nbin_lens = n

        for i in range(nbin_source):
            norm = np.trapezoid(source_nz[i], source_z)
            if norm <= 0:
                raise ValueError("LSST source n(z) has non-positive normalization")
            source_nz[i] = source_nz[i] / norm
            source_nz[i] *= source_number_densities[i]

        for i in range(nbin_lens):
            norm = np.trapezoid(lens_nz[i], lens_z)
            if norm <= 0:
                raise ValueError("LSST lens n(z) has non-positive normalization")
            lens_nz[i] = lens_nz[i] / norm
            lens_nz[i] *= lens_number_densities[i]

        # If combined=True, compress to a single bin consistently.
        if combined:
            source_nz = np.array([np.sum(source_nz, axis=0)])
            lens_nz = np.array([np.sum(lens_nz, axis=0)])
            nbin_source = 1
            nbin_lens = 1

        sample = SampleInfo()
        sample.nbin_source = nbin_source
        sample.nbin_lens = nbin_lens
        sample.source_z = source_z
        sample.source_nz = source_nz
        sample.lens_z = lens_z
        sample.lens_nz = lens_nz
        sample.lens_number_densities = lens_number_densities
        sample.galaxy_bias = galaxy_bias
        sample.source_number_densities = source_number_densities
        sample.sigma_e = sigma_e
        return sample

    source_data = np.loadtxt(source_nz_filename).T

    source_z = source_data[0]
    source_nz = source_data[1:]

    if combined:
        source_number_densities = [combined_source_number_densities]
        sigma_e = combined_sigma_e
    else:
        source_number_densities = tomographic_source_number_densities
        sigma_e = tomographic_sigma_e


    if lens_type == "maglim":
        if combined:
            lens_number_densities = [maglim_combined_number_densities]
            galaxy_bias = [np.mean(combined_maglim_bias)]
            lens_data = np.loadtxt(combined_maglim_nz_filename).T
        else:
            sample.lens_number_densities = tomographic_maglim_number_densities
            sample.galaxy_bias = tomographic_maglim_bias
            lens_data = np.loadtxt(tomographic_maglim_nz_filename).T
            
    else:
        if combined:
            lens_number_densities = [redmagic_combined_number_densities]
            galaxy_bias = np.mean(tomographic_redmagic_bias)
            lens_data = np.loadtxt(combined_redmagic_nz_filename).T
        else:
            lens_number_densities = tomographic_redmagic_number_densities
            galaxy_bias = tomographic_redmagic_bias
            lens_data = np.loadtxt(tomographic_redmagic_nz_filename).T

    lens_z = lens_data[0]
    lens_nz = lens_data[1:]
    nbin_lens = len(lens_nz)
    nbin_source = len(source_nz)

    for i in range(nbin_source):
        source_nz[i] /= np.trapezoid(source_nz[i], source_z)
        source_nz[i] *= source_number_densities[i]
    
    for i in range(nbin_lens):
        lens_nz[i] /= np.trapezoid(lens_nz[i], lens_z)
        lens_nz[i] *= lens_number_densities[i]


    sample = SampleInfo()
    sample.nbin_source = nbin_source
    sample.nbin_lens = nbin_lens
    sample.source_z = source_z
    sample.source_nz = source_nz
    sample.lens_z = lens_z
    sample.lens_nz = lens_nz
    sample.lens_number_densities = lens_number_densities
    sample.galaxy_bias = galaxy_bias
    sample.source_number_densities = source_number_densities
    sample.sigma_e = sigma_e


    return sample























def load_mask(nside, lsst=False):
    if lsst:
        if lsst not in [1, 10]:
            raise ValueError("lsst must be 1 or 10 if not False/0")
    if lsst:
        input_mask_nside = 64
    else:
        input_mask_nside = 4096

    if lsst == 1:
        mask = healpy.read_map(lsst_mask_filename_y1, verbose=False)
    elif lsst == 10:
        mask = healpy.read_map(lsst_mask_filename_y10, verbose=False)
    else:
        hit_pix = np.load(mask_filename)
        mask = np.zeros(healpy.nside2npix(input_mask_nside))
        mask[hit_pix] = 1
        mask = healpy.reorder(mask, n2r = True)

    # degrade mask to nside quality of simulation
    mask = healpy.ud_grade(mask, nside_out = nside)
    return mask

