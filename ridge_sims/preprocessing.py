import numpy as np
import h5py
from astropy.io import fits
from astropy.table import Table

gold_mask_file = "DESY3_GOLD_2_2.1.h5"
redmagic_nz_file = "2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits"
maglim_nz_file = "2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits"

def calibrate_shears(index_file, metacal_file):
    delta_gamma = 0.02

    with h5py.File(index_file, 'r') as f:
        # for the selection bias calculation.
        # These are index arrays into the full set.
        s00 = f["/index/metacal/select"][:]
        s1m = f["/index/metacal/select_1m"][:]
        s1p = f["/index/metacal/select_1p"][:]
        s2m = f["/index/metacal/select_2m"][:]
        s2p = f["/index/metacal/select_2p"][:]


    with h5py.File(metacal_file, 'r') as f:
        # The main response term is just the mean of the Rij columns
        # selected by the s00
        R11 = f['catalog/unsheared/R11'][:][s00].mean()
        R22 = f['catalog/unsheared/R22'][:][s00].mean()
        R12 = f['catalog/unsheared/R12'][:][s00].mean()
        R21 = f['catalog/unsheared/R21'][:][s00].mean()
        R_gamma = np.array([[R11, R12], [R21, R22]])

        e1 = f['catalog/unsheared/e_1'][:]
        e2 = f['catalog/unsheared/e_2'][:]


    S11 = (e1[s1p].mean() - e1[s1m].mean()) / delta_gamma
    S22 = (e2[s2p].mean() - e2[s2m].mean()) / delta_gamma
    S12 = (e1[s2p].mean() - e1[s2m].mean()) / delta_gamma
    S21 = (e2[s1p].mean() - e2[s1m].mean()) / delta_gamma
    R_S = np.array([[S11, S12], [S21, S22]])

    R = R_gamma + R_S

    R_inv = np.linalg.inv(R)
    e1, e2 = R_inv @ [e1, e2]

    return s00, e1, e2



def extract_source_samples(index_file, metacal_file, shear_output_file):
    # 1 load the metacal sample, apply the /index/metacal/select selection,
    # calibrate it (R and S), and save it.

    sel, e1, e2 = calibrate_shears(index_file, metacal_file)
    with h5py.File(index_file, 'r') as f:
        ra = f['/index/metacal/ra'][:][sel]
        dec = f['/index/metacal/dec'][:][sel]
        weight = f['/index/metacal/weight'][:][sel]

    with h5py.File(shear_output_file, "w") as f:
        f.create_dataset("ra", data=ra)
        f.create_dataset("dec", data=dec)
        f.create_dataset("e1", data=e1)
        f.create_dataset("e2", data=e2)
        f.create_dataset("weight", data=weight)
    
    # we can just use the source n(z) file for this since it should match,
    # as we are not doing any additional cuts, so no need to extract it from anywhere

def extract_maglim_sample(index_file, lens_file, dnf_file, maglim_output_file):
    with h5py.File(index_file, 'r') as f:
        sel = f["/index/maglim/select"][:]

    with h5py.File(lens_file, "r") as f:
        ra = f["/catalog/maglim/ra"][sel]
        dec = f["/catalog/maglim/dec"][sel]
        weight = f["/catalog/maglim/weight"][sel]
    
    with h5py.File(dnf_file, "r") as f:
        # used for estimating the ensemble
        z_mc = f["/catalog/unsheared/zmc_sof"][sel]
        # used for the cut
        z_mean = f["/catalog/unsheared/zmean_sof"][sel]

    with h5py.File(maglim_output_file, "w") as f:
        f.create_dataset("ra", data=ra)
        f.create_dataset("dec", data=dec)
        f.create_dataset("weight", data=weight)
        f.create_dataset("z_sample", data=z_mc)
        f.create_dataset("z", data=z_mean)


def extract_redmagic_sample(index_file, lens_file, redmagic_output_file):

    with h5py.File(index_file, 'r') as f:
        sel = f["/index/redmagic/combined_sample_fid/select"][:]

    with h5py.File(lens_file, "r") as f:
        ra = f["/catalog/redmagic/combined_sample_fid/ra"][sel]
        dec = f["/catalog/redmagic/combined_sample_fid/dec"][sel]
        weight = f["/catalog/redmagic/combined_sample_fid/weight"][sel]
        z = f["/catalog/redmagic/combined_sample_fid/zredmagic"][sel]
        z_sample = f["/catalog/redmagic/combined_sample_fid/zredmagic_samp"][sel]

    with h5py.File(redmagic_output_file, "w") as f:
        f.create_dataset("ra", data=ra)
        f.create_dataset("dec", data=dec)
        f.create_dataset("weight", data=weight)
        f.create_dataset("z_sample", data=z_sample)
        f.create_dataset("z", data=z)

def estimate_lens_nz_with_cut(input_file, zmax, output_file):
    with h5py.File(input_file) as f:
        z = f["/catalog/unsheared/z"][:]
        weight = f["/catalog/unsheared/weight"][:]
        z_mc = f["/catalog/unsheared/zmc"][:]

    cut = z < zmax
    z_mc = z_mc[cut]
    weight = weight[cut]

    dz = 0.01
    counts, edges = np.histogram(z_mc, weights=weight, bins=np.arange(0, zmax+dz/2, dz))
    mids = 0.5 * (edges[1:] + edges[:-1])
    np.savetxt(output_file, np.transpose([mids, counts]), header="z n_z")




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
