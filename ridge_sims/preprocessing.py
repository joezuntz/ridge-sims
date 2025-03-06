import numpy as np
import h5py
from astropy.io import fits
from astropy.table import Table

gold_mask_file = "DESY3_GOLD_2_2.1.h5"
redmagic_nz_file = "2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits"
maglim_nz_file = "2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits"

def extract_des_mask_from_gold(mask_file):
    """
    This is a one-off pre-run step to pull
    the DES Y3 mask from the gold mask file.

    Once it's done you can just load the mask
    from the numpy mask file.
    """
    with h5py.File(gold_mask_file, 'r') as f:
        # This is an index of hea
        mask = f["/masks/gold/hpix"][:]
    np.save(mask, mask_file)

def extract_nz(nz_fits_file, lens_output_file, source_output_file):
    """
    Extract the redmagic and maglim n(z) from the
    respective files.
    """
    source_extname = "nz_source"
    lens_extname = "nz_lens"
    nbin_source = 4

    if "maglim" in lens_output_file:
        nbin_lens = 6
    else:
        nbin_lens = 5
    with fits.open(nz_fits_file) as f:
        source_data = f[source_extname].data
        lens_data = f[lens_extname].data
    
        source_z = source_data["Z_MID"]
        nz_source = [source_data[f"BIN{i}"] for i in range(1, nbin_source+1)]

        source_table = Table([source_z] + nz_source, names=["Z"] + [f"BIN{i}" for i in range(1, nbin_source+1)])
        source_table.write(source_output_file, format='ascii.commented_header', overwrite=True)

        lens_z = lens_data["Z_MID"]
        nz_lens = [lens_data[f"BIN{i}"] for i in range(1, nbin_lens+1)]

        lens_table = Table([lens_z] + nz_lens, names=["Z"] + [f"BIN{i}" for i in range(1, nbin_lens+1)])
        lens_table.write(lens_output_file, format='ascii.commented_header', overwrite=True)

def extract_source_nz(nz_fits_file, source_output_file):
    source_extname = "nz_source"
    nbin_source = 4
    with fits.open(nz_fits_file) as f:
        source_data = f[source_extname].data    
        source_header = f[source_extname].header
        source_z = source_data["Z_MID"]
        nz_source = [source_data[f"BIN{i}"] for i in range(1, nbin_source+1)]
        nz_ngal = [source_header[f'NGAL_{i}'] for i in range(1, nbin_source+1)]
        nz_source_combined =  sum([nz_source[i] * nz_ngal[i] for i in range(nbin_source)])

    cut = source_z < 2.0
    source_z = source_z[cut]
    nz_source_combined = nz_source_combined[cut]
    with open(source_output_file, 'w') as f:
        f.write("# z n(z)\n")
        for z, nz in zip(source_z, nz_source_combined):
            f.write(f"{z} {nz}\n")


def extract_all_nz():
    # The source n(z) is the same in the two files so the overwrite doesn't matter here
    extract_nz(redmagic_nz_file, "redmagic_nz.txt", "source_nz.txt")
    extract_nz(maglim_nz_file, "maglim_nz.txt", "source_nz.txt")


if __name__ == "__main__":
    # extract_all_nz()
    extract_source_nz(maglim_nz_file, "source_nz_combined.txt")
