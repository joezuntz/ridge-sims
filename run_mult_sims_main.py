import os
import sys
import numpy as np
from ridge_sims.steps import step1, step2, step3
from ridge_sims.config import Config

if __name__ == "__main__":
    # --- Configuration for direct execution ---
    if len(sys.argv) == 1:  # run without arguments
        num_runs = 10
        base_sim_dir = "local_lhc_run"
        fiducial_sim_dir = "sim-fiducial"  # might need future adjustments for local runs
        fiducial_shell_cl_file = os.path.join(fiducial_sim_dir, "shell_cls.npy")

        for run_id in range(1, num_runs + 1):
            run_dir = os.path.join(base_sim_dir, f"run_{run_id}")
            seed = run_id
            np.random.seed(seed)

            config = Config(sim_dir=run_dir, seed=seed)
            config.save()

            print(f"Starting local run {run_id} in directory: {run_dir} with seed: {seed}")

            if not os.path.exists(fiducial_shell_cl_file):
                print(f"Fiducial shell_cl file not found at {fiducial_shell_cl_file}. Running step 1.")
                step1(config)
            else:
                print(f"Fiducial shell_cl file found. Skipping step 1 for local run {run_id}.")

            config.shell_cl_file = fiducial_shell_cl_file
            step2(config)
            step3(config)

            print(f"Finished local run {run_id}")

    # --- Configuration for Slurm execution ---
    elif len(sys.argv) == 4:
        run_dir = sys.argv[1]
        run_id = int(sys.argv[2])
        fiducial_shell_cl_file = sys.argv[3]

        seed = run_id
        np.random.seed(seed)

        config = Config(sim_dir=run_dir)
        config.save()

        print(f"Starting Slurm run {run_id} in directory: {run_dir} with seed: {seed}")

        if not os.path.exists(fiducial_shell_cl_file):
            print(f"Fiducial shell_cl file not found at {fiducial_shell_cl_file}. Running step 1.")
            step1(config)
        else:
            print(f"Fiducial shell_cl file found. Skipping step 1 for Slurm run {run_id}.")

        config.shell_cl_file = fiducial_shell_cl_file
        step2(config)
        step3(config)

        print(f"Finished Slurm run {run_id}")

    else:
        print("Usage:")
        print("  python main.py                      (to run 10 local simulations)")
        print("  python main.py <run_dir> <run_id> <fiducial_shell_cl> (for Slurm)")
        sys.exit(1)