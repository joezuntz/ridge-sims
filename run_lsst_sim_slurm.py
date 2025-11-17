import argparse
import os
import numpy as np
from ridge_sims.steps import step1, step2, step3
from ridge_sims.config import Config

def map_job_to_config(job_id):
    """
    Map a single SLURM job_id to (include_shape_noise, run_id)
    
    Example:
    job_id:   1 2 3 4 5 6
    noise :   T T T F F F
    run_id:   1 2 3 1 2 3
    """
    # runs per mode
    runs_per_mode = 3

    if job_id <= runs_per_mode:
        include_noise = True
        run_id = job_id
    else:
        include_noise = False
        run_id = job_id - runs_per_mode

    return include_noise, run_id


def run_sim(run_id, include_noise):

    base_dir = (
        "lhc_run_lsst_sims" if include_noise 
        else "lhc_run_lsst_sim_zero_err"
    )

    run_dir = os.path.join(base_dir, f"run_{run_id}")
    seed = run_id
    np.random.seed(seed)

    # First run path
    first_run_dir = os.path.join(base_dir, "run_1")
    first_shell = os.path.join(first_run_dir, "shell_cls.npy")

    # Create config
    config = Config(
        sim_dir=run_dir,
        seed=seed,
        include_shape_noise=include_noise,
        lsst=True,
    )
    config.save()

    print(f"\n=== include_shape_noise={include_noise} | run {run_id} ===")

    # Step 1 logic... Do not run this step for the next runs
    if run_id == 1:
        print("Running step 1 for run 1")
        step1(config)
    else:
        if os.path.exists(first_shell):
            print("Reusing shell_cls.npy from run 1")
            config.shell_cl_file = first_shell
        else:
            print("shell_cls.npy missing → recomputing")
            step1(config)
            config.shell_cl_file = first_shell

    # Step 2 + Step 3
    step2(config)
    step3(config)

    print(f"Finished run {run_id} (include_noise={include_noise})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, required=True)
    args = parser.parse_args()

    include_noise, run_id = map_job_to_config(args.job_id)

    print(f"Dispatcher assigned:")
    print(f"  include_shape_noise = {include_noise}")
    print(f"  run_id              = {run_id}")

    run_sim(run_id, include_noise)
