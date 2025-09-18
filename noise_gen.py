import os
import h5py
from ridge_analysis_tools import transform_background

def main():
    base_sim_dir = "lhc_run_sims_50"
    num_simulations = 8
    num_realizations_per_sim = 300
    
    # Central directory for all noise files
    output_dir = "example30_band0.4/8test/noise_data"
    os.makedirs(output_dir, exist_ok=True)
    
    for run_id in range(1, num_simulations + 1):
        print(f"--- Generating noise for Run {run_id} ---")
        bg_data_path = os.path.join(base_sim_dir, f"run_{run_id}", "source_catalog_cutzl04.h5")
        
        for i in range(num_realizations_per_sim):
            seed = (run_id - 1) * num_realizations_per_sim + i
            output_file_path = os.path.join(output_dir, f"noise_r{run_id:02d}_n{i:03d}.h5")
            transform_background(bg_data_path, output_file_path, seed=seed)
            if i % 50 == 0:
                print(f"  - Generated noise {i} for run {run_id}")

if __name__ == "__main__":
    main()