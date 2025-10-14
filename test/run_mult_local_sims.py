import os
import sys
import numpy as np
from ridge_sims.steps import step1, step2, step3
from ridge_sims.config import Config

if __name__ == "__main__":
    # --- Configuration for direct execution ---
    if len(sys.argv) == 1:  # run without arguments
        num_runs = 10
        base_sim_dir = "lhc_run_sims_zero_err_10"
        first_run_dir = os.path.join(base_sim_dir, "run_1")
        first_run_shell_cl_file = os.path.join(first_run_dir, "shell_cls.npy")

        for run_id in range(1, num_runs + 1):
            run_dir = os.path.join(base_sim_dir, f"run_{run_id}")
            seed = run_id
            np.random.seed(seed)

            config = Config(sim_dir=run_dir, seed=seed, include_shape_noise=False)
            config.save()

            print(f"Starting local run {run_id} in directory: {run_dir} with seed: {seed}")

            if run_id == 1:
                # For the first run, always execute step 1
                print(f"Running step 1 for the first run.")
                step1(config)
                config.shell_cl_file = first_run_shell_cl_file
            else:
                # For subsequent runs, check if the shell_cl file from the first run exists
                if os.path.exists(first_run_shell_cl_file):
                    print(f"Found shell_cl file from the first run at {first_run_shell_cl_file}. Skipping step 1.")
                    config.shell_cl_file = first_run_shell_cl_file
                else:
                    print(f"Warning: shell_cl file from the first run not found. Running step 1 for run {run_id}.")
                    step1(config)
                    config.shell_cl_file = first_run_shell_cl_file # Still point to the expected location

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
            config.shell_cl_file = os.path.join(run_dir, "shell_cls.npy")  # Update path after running step 1
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