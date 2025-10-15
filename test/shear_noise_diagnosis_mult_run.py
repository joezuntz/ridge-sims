"""
Shear calculation for all runs and bands.

Produces, per bandwidth:
  shear_calc/
    raw_ridges_shear/
      filament_segments/
      noise_shear/
    contracted_ridges_shear/
      filament_segments/
      noise_shear/
.
"""

import os, sys
import re
import time
import glob
import numpy as np
import h5py
from ridge_analysis_tools import * 
from mpi4py import MPI

comm = MPI.COMM_WORLD


############################################################
############################################################

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())


############################################################
############################################################



# ------------------------
# Config
# ------------------------
base_root = "simulation_ridges_comparative_analysis_test"   # Base folder: zero_err/normal folders
noise_dir = "example_zl04_mesh5e5/noise"               # Noise realizations folder
final_percentiles = [15]                               # percentiles to process
nside_for_plots = 512

#  base_label -> Simulation directory for BG catalogs
sim_bases_map = {
    "zero_err": "lhc_run_sims_zero_err_10",
    "normal": "lhc_run_sims"
}

# background catalog template -- uses run_id
bg_filename_template = "source_catalog_cutzl04.h5"  

# ------------------------
# Utilities
# ------------------------
def extract_run_id_from_filename(filename):
    """
    Extract numeric run id from filenames like:
      zero_err_run_1_ridges_p15.h5  or  normal_run_03_ridges_p15.h5
    Returns integer run_id or None if not found.
    """
    m = re.search(r"run[_\-]?(\d+)", filename)
    if m:
        return int(m.group(1))
    # fallback: try trailing digits    
    m2 = re.search(r"(\d+)\.h5$", filename)
    if m2:
        return int(m2.group(1))
    return None

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# ------------------------
# Main processing for one ridge file
# ------------------------
def process_single_ridge_file(ridge_file, base_label, band_path, run_id, fp):
    """
    ridge_file: path to original ridge .h5 (raw or contracted)
    base_label: "zero_err" or "normal"
    band_path: full path to the band directory (e.g., base_root/zero_err/band_0.1)
    run_id: integer run id
    fp: final percentile integer (15)
    """
    # Setup shear_calc directories: for ref: directly under band_path 
    shear_calc_dir = os.path.join(band_path, "shear_calc")
    raw_dir = os.path.join(shear_calc_dir, "raw_ridges_shear")
    contracted_dir = os.path.join(shear_calc_dir, "contracted_ridges_shear")

    # Determine whether this ridge_file is raw or contracted by parent folder name    
    parent_folder = os.path.basename(os.path.dirname(ridge_file))
    is_contracted = "contracted" in os.path.basename(os.path.dirname(ridge_file)) or "_contracted" in os.path.basename(ridge_file)

    target_group = "contracted_ridges_shear" if is_contracted else "raw_ridges_shear"
    target_base = contracted_dir if is_contracted else raw_dir

    filament_segments_dir = os.path.join(target_base, "filament_segments")
    noise_shear_dir = os.path.join(target_base, "noise_shear")

    ensure_dirs(filament_segments_dir, noise_shear_dir)

    # Filament HDF5 output path    
    filaments_h5 = os.path.join(filament_segments_dir, f"filaments_run{run_id}_p{fp:02d}.h5")

    # Only rank 0 performs the filament segmentation & saves HDF5    
    if comm is None or comm.rank == 0:
        print(f"[rank 0] Loading ridges from {ridge_file}")
        with h5py.File(ridge_file, "r") as f:
            ridges = f["ridges"][:]

        # Build MST, detect branches and segment        
        print(f"[rank 0] Building MST and segmenting (run {run_id}, fp={fp}) ...")
        mst = build_mst(ridges)
        branch_points = detect_branch_points(mst)
        filament_segments = split_mst_at_branches(mst, branch_points)
        filament_labels = segment_filaments_with_dbscan(ridges, filament_segments)

        # Save filaments into filaments_h5        
        save_filaments_to_hdf5(ridges, filament_labels, filaments_h5)
        print(f"[rank 0] Saved filaments -> {filaments_h5}")

    # sync     
    if comm is not None:
        comm.Barrier()

    ####### Compute shear ##############    
    # Background data path for this run:    
    base_sim_dir = sim_bases_map.get(base_label, base_label)
    bg_file = os.path.join(base_sim_dir, f"run_{run_id}", bg_filename_template)

    # Shear output files (signal)    
    shear_csv = os.path.join(filament_segments_dir, f"shear_run{run_id}_p{fp:02d}.csv")
    shear_flip_csv = os.path.join(filament_segments_dir, f"shear_run{run_id}_p{fp:02d}_flipG1.csv")

    # run on signal (background)    
    process_shear_sims(filaments_h5, bg_file, output_shear_file=shear_csv, background_type='sim', plot_output_dir=filament_segments_dir)
    # flipped signal    
    process_shear_sims(filaments_h5, bg_file, output_shear_file=shear_flip_csv, flip_g1=True, background_type='sim')

    # === Loop over noise realizations ===    
    # === Automatically select a subset of noise realizations ===    
    # You can control how many to take:    
    n_realizations_to_use = 50 
    start_realization = 1       # or set to any number to start from there    
    
    # Find all available noise files    
    all_noise_files = sorted(
        [f for f in os.listdir(noise_dir)
         if re.match(r"source_catalog_noise_\d+\.h5", f)],
        key=lambda x: int(re.search(r"(\d+)\.h5", x).group(1))
    )
    
    # Extract realization numbers    
    all_ids = [int(re.search(r"(\d+)\.h5", f).group(1)) for f in all_noise_files]
    
    # Filter based on start index and limit how many to use    
    selected_ids = [rid for rid in all_ids if rid >= start_realization][:n_realizations_to_use]
    
    # Build list of existing files    
    noise_files = [
        f"source_catalog_noise_{rid}.h5"
        for rid in selected_ids
        if os.path.exists(os.path.join(noise_dir, f"source_catalog_noise_{rid}.h5"))
    ]
    
    print(f"Processing {len(noise_files)} noise realizations: {selected_ids}")

    all_noise_profiles = []
    all_noise_flip_profiles = []

    # loop over noise
    for nf in noise_files:
        try:
            realization_id = int(nf.split("_")[-1].replace(".h5", ""))
            # Use 3 digits for consistent file naming, e.g., "001"
            realization_id_str = f"{realization_id:03d}" 
        except ValueError:
             realization_id_str = nf.split("_")[-1].replace(".h5", "") 

        noise_file = os.path.join(noise_dir, nf)

        # Output shear files per realization    
        shear_noise_csv_i = os.path.join(noise_shear_dir, f"shear_noise_p{fp:02d}_{realization_id_str}.csv")
        shear_noise_flip_csv_i = os.path.join(noise_shear_dir, f"shear_noise_p{fp:02d}_{realization_id_str}_flipG1.csv")

        # Compute shear for this noise realization
        
        process_shear_sims(filaments_h5, noise_file, output_shear_file=shear_noise_csv_i, background_type='sim')
        process_shear_sims(filaments_h5, noise_file, output_shear_file=shear_noise_flip_csv_i,
                           flip_g1=True, background_type='sim')

        # Load into memory     
        all_noise_profiles.append(np.loadtxt(shear_noise_csv_i, delimiter=",", skiprows=1))
        all_noise_flip_profiles.append(np.loadtxt(shear_noise_flip_csv_i, delimiter=",", skiprows=1))

    # Convert to arrays (N_realizations, N_bins, N_columns)    
    all_noise_profiles = np.array(all_noise_profiles)
    all_noise_flip_profiles = np.array(all_noise_flip_profiles)

    # Mean across realizations    
    mean_noise = np.mean(all_noise_profiles, axis=0)
    mean_noise_flip = np.mean(all_noise_flip_profiles, axis=0)

    # === Subtract mean noise from signal ===    
    if comm is None or comm.rank == 0:
        print(f"Subtracting mean noise from signal")

        shear_data = np.loadtxt(shear_csv, delimiter=",", skiprows=1)
        shear_data_flip = np.loadtxt(shear_flip_csv, delimiter=",", skiprows=1)

        g_plus_subtracted = shear_data[:, 2] - mean_noise[:, 2]
        g_cross_subtracted = shear_data[:, 3] - mean_noise[:, 3]

        g_plus_subtracted_flip = shear_data_flip[:, 2] - mean_noise_flip[:, 2]
        g_cross_subtracted_flip = shear_data_flip[:, 3] - mean_noise_flip[:, 3]

        
        
        # Save subtracted signal        
        subtracted_data = np.column_stack((
            shear_data[:, 0],  # Bin_Center            
            shear_data[:, 1],  # Weighted_Real_Distance            
            g_plus_subtracted,
            g_cross_subtracted,
            shear_data[:, 4],  # Counts            
            shear_data[:, 5]   # bin_weight        
        ))
        subtracted_output_file = os.path.join(
            filament_segments_dir, f"shear_p{fp:02d}_shear-randomshear.csv"
        )
        np.savetxt(
            subtracted_output_file,
            subtracted_data,
            delimiter=",",
            header="Bin_Center,Weighted_Real_Distance,Weighted_g_plus_subtracted,Weighted_g_cross_subtracted,Counts,bin_weight",
            comments=""
        )
        print(f"Saved mean-noise-subtracted shear to {subtracted_output_file}")

        # Save flipped subtracted signal        
        subtracted_data_flip = np.column_stack((
            shear_data_flip[:, 0],
            shear_data_flip[:, 1],
            g_plus_subtracted_flip,
            g_cross_subtracted_flip,
            shear_data_flip[:, 4],
            shear_data_flip[:, 5]
        ))
        subtracted_output_file_flip = os.path.join(
            filament_segments_dir, f"shear_p{fp:02d}_flipG1_shear-randomshear.csv"
        )
        np.savetxt(
            subtracted_output_file_flip,
            subtracted_data_flip,
            delimiter=",",
            header="Bin_Center,Weighted_Real_Distance,Weighted_g_plus_subtracted,Weighted_g_cross_subtracted,Counts,bin_weight",
            comments=""
        )
        print(f"Saved mean-noise-subtracted flipped shear to {subtracted_output_file_flip}")

    if comm is not None:
        comm.Barrier()  # ensure everyone syncs before next file

## ------------------------
## bandwidths & process files -> Test for only One case 
## ------------------------
def main():
    # Select run config    
    base_label = "zero_err"
    bandwidth = 0.1
    run_id = 1
    fp = 15  # final percentile

    # The band_path is the base directory for the entire bandwidth run
    band_path = os.path.join(base_root, base_label, f"band_{bandwidth}")
    
    # Build path to the ridge file (contracteded) -> This is the INPUT FILE location
    ridge_dir = os.path.join(band_path, f"contracted_Ridges_final_p{fp}")
    ridge_file = os.path.join(ridge_dir, f"{base_label}_run_{run_id}_ridges_p{fp}_contracted.h5")

    if not os.path.exists(ridge_file):
        raise FileNotFoundError(f" Ridge file not found: {ridge_file}")

    print(f"   Running for: {ridge_file}")
    print(f"   base_label = {base_label}")
    print(f"   bandwidth = {bandwidth}")
    print(f"   run_id = {run_id}")
    print(f"   percentile = {fp}")

    # Run the main processing on the contracted ridges    
    # NOTE: band_path is passed as the root for creating the shear_calc directory.
    process_single_ridge_file(ridge_file, base_label, band_path, run_id, fp)

if __name__ == "__main__":
    main()














## ------------------------
## bandwidths & process files -> This is to do everything at the sametime 
## ------------------------
#def main():
#    t0 = time.time()

#    # iterate base labels folders inside base_root
#    for base_label in sorted(os.listdir(base_root)):
#        base_path = os.path.join(base_root, base_label)
#        if not os.path.isdir(base_path):
#            continue

#        # map to simulation base dir for BG catalogs
#        sim_base_dir = sim_bases_map.get(base_label, None)
#        if sim_base_dir is None:
#            print(f"[WARN] No sim base mapping for {base_label}; skipping")
#            continue

#        # for each band folder
#        for band_folder in sorted(os.listdir(base_path)):
#            if not band_folder.startswith("band_"):
#                continue
#            band_path = os.path.join(base_path, band_folder)

#            # Look for both raw and contracted ridge folders
#            ridge_folders = []
#            raw_ridges_dir = os.path.join(band_path, "Ridges_final_p15")
#            if os.path.isdir(raw_ridges_dir):
#                ridge_folders.append(raw_ridges_dir)
#            contracted_ridges_dir = os.path.join(band_path, "contracted_Ridges_final_p15")
#            if os.path.isdir(contracted_ridges_dir):
#                ridge_folders.append(contracted_ridges_dir)

#            if len(ridge_folders) == 0:
#                continue

#            # Create shear_calc root
#            shear_calc_root = os.path.join(band_path, "shear_calc")
#            if comm is None or comm.rank == 0:
#                os.makedirs(shear_calc_root, exist_ok=True)

#            if comm is not None:
#                comm.Barrier()

#            print(f"\n[INFO] Processing base={base_label} band={band_folder}, ridge_dirs={ridge_folders}")

#            # Process each ridge folder (raw or contracted)
#            for rdir in ridge_folders:
#                # list ridge files
#                ridge_files = sorted([f for f in os.listdir(rdir) if f.endswith(".h5")])
#                if len(ridge_files) == 0:
#                    print(f"[INFO] No ridge files in {rdir}; skipping")
#                    continue

#                # For each ridge file run the processing
#                for rf in ridge_files:
#                    full_ridge_path = os.path.join(rdir, rf)
#                    run_id = extract_run_id_from_filename(rf)
#                    if run_id is None:
#                        print(f"[WARN] Could not parse run_id from {rf}; skipping")
#                        continue

#                    for fp in final_percentiles:
#                        # process (this includes filament segmentation, shear, noise shear, subtraction)
#                        process_single_ridge_file(full_ridge_path, base_label, band_path, run_id, fp)

#    t1 = time.time()
#    if comm is None or comm.rank == 0:
#        print(f"\nAll done. Total time: {t1 - t0:.1f} s")



  