import numpy as np
import healpy
from types import SimpleNamespace

class SampleInfo(SimpleNamespace):
    pass

mask_filename = "des-data/desy3_gold_mask.npy"

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



def load_sample_information(lens_type, combined=True):
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

    Returns
    -------
    sample : SampleInfo
        The object containing the number density information.
    """

    source_data = np.loadtxt(source_nz_filename).T

    source_z = source_data[0]
    source_nz = source_data[1:]

    if combined:
        source_number_densities = [combined_source_number_densities]
        source_nz = [tomographic_source_number_densities @ source_nz]
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






def load_mask(nside):
    input_mask_nside = 4096
    hit_pix = np.load(mask_filename)
    mask = np.zeros(healpy.nside2npix(input_mask_nside))
    mask[hit_pix] = 1
    # degrade mask to nside quality of simulation
    mask = healpy.reorder(mask, n2r = True)
    mask = healpy.ud_grade(mask, nside_out = nside)
    return mask

