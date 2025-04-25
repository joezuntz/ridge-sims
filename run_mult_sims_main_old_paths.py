import os
import sys
import numpy as np
from ridge_sims.steps import step1, step2, step3
from ridge_sims.config import Config

if __name__ == "__main__":
    # --- Configuration for direct execution ---
    if len(sys.argv) == 1:  # run without arguments
        num_runs = 10
        base_sim_dir = "local_lhc_run_0"
        fiducial_shell_cl_file_local = "/home/jzuntz/shell_cls.npy"  # Explicit path for local runs
        local_g_ell_file = "/home/jzuntz/g_ell.pkl"  # Explicit path for local runs

        for run_id in range(1, num_runs + 1):
            run_dir = os.path.join(base_sim_dir, f"run_{run_id}")
            seed = run_id
            np.random.seed(seed)

            config = Config(sim_dir=run_dir, seed=seed)
            config.save()

            print(f"Starting local run {run_id} in directory: {run_dir} with seed: {seed}")

            if not os.path.exists(fiducial_shell_cl_file_local):
                print(f"Fiducial shell_cl file not found at {fiducial_shell_cl_file_local}. Running step 1.")
                step1(config)
            else:
                print(f"Fiducial shell_cl file found at {fiducial_shell_cl_file_local}. Skipping step 1 for local run {run_id}.")

            config.shell_cl_file = fiducial_shell_cl_file_local  # Use the explicit path
            # You might need to adjust how g_ell is used in step2.
            # If step2 expects to *create* g_ell, you might not want to load it here.
            # If step2 expects to *load* g_ell, you can set config.g_ell_file here.
            # Example (assuming config has a g_ell_file attribute):
            # config.g_ell_file = local_g_ell_file

            step2(config)
            step3(config)

            print(f"Finished local run {run_id}")

    # --- Configuration for Slurm execution ---
    elif len(sys.argv) == 4:
        run_dir = sys.argv[1]
        run_id = int(sys.argv[2])
        fiducial_shell_cl_file_slurm = sys.argv[3]  # Path from Slurm

        seed = run_id
        np.random.seed(seed)

        config = Config(sim_dir=run_dir)
        config.save()

        print(f"Starting Slurm run {run_id} in directory: {run_dir} with seed: {seed}")

        if not os.path.exists(fiducial_shell_cl_file_slurm):
            print(f"Fiducial shell_cl file not found at {fiducial_shell_cl_file_slurm}. Running step 1.")
            step1(config)
        else:
            print(f"Fiducial shell_cl file found at {fiducial_shell_cl_file_slurm}. Skipping step 1 for Slurm run {run_id}.")

        config.shell_cl_file = fiducial_shell_cl_file_slurm  # Use the path from Slurm
        step2(config)
        step3(config)

        print(f"Finished Slurm run {run_id}")

    else:
        print("Usage:")
        print("  python main.py                      (to run 10 local simulations)")
        print("  python main.py <run_dir> <run_id> <fiducial_shell_cl> (for Slurm)")
        sys.exit(1)