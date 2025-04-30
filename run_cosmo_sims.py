# run_different_cosmo_direct.py
import os
import numpy as np
from ridge_sims.steps import step1, step2, step3
from ridge_sims.config import Config

if __name__ == "__main__":
    num_processes = 4  # You can adjust this based on your local machine's resources
    base_sim_dir = "cosmo_direct_runs"

    def run_cosmology(i):
        sigma8 = 0.8 + (i - 5) * 0.01
        Omega_m = 0.3 + (i - 5) * 0.01
        run_id = f"run_{i+1}"
        sim_dir = os.path.join(base_sim_dir, run_id)

        config = Config(
            Omega_m=Omega_m,
            sigma8=sigma8,
            sim_dir=sim_dir,
            seed=i,
        )
        config.save()

        print(f"Starting cosmology run {i+1} with Omega_m={Omega_m:.3f}, sigma8={sigma8:.3f} in directory: {sim_dir} with seed: {i}")

        step1(config)
        step2(config)
        step3(config)

        print(f"Finished cosmology run {i+1}")

    # Run the cosmologies sequentially
    for i in range(10):
        run_cosmology(i)

    print("All cosmology runs completed.")