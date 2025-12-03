# We want the following file structure to be created 

"""
lhc_run_lsst_sims/
    lsst_1/
        run_1/
        run_2/
        
    lsst_10/
        run_1/
        run_2/
        
lhc_run_lsst_sim_zero_err/
    lsst_1/
        run_1/
        ...
    lsst_10/
        run_1/
        ...
"""






import argparse
import os
import numpy as np
from ridge_sims.steps import step1, step2, step3
from ridge_sims.config import Config

def map_job_to_config(job_id):
    """
    job_id → (include_noise, run_id, lsst)
    """
    if not 1 <= job_id <= 8:
        raise ValueError("job_id must be 1–8")

    # lsst mode
    lsst = 1 if job_id <= 4 else 10

    # reduce to 1..4 for noise/run logic
    local_id = job_id if job_id <= 4 else job_id - 4

    # noise / run split
    include_noise = (local_id <= 2)   # first two → noise
    run_id = 1 if local_id in (1,3) else 2

    return include_noise, run_id, lsst





def map_missing_job_to_config(job_id):
    """
    Missing simulations for LSST=10 only.
    job_id:
        1 → noise, run 1
        2 → noise, run 2
        3 → zero-noise, run 1
        4 → zero-noise, run 2
    """
    if not 1 <= job_id <= 4:
        raise ValueError("job_id must be 1–4")

    include_noise = (job_id <= 2)
    run_id = 1 if job_id in (1, 3) else 2

    return include_noise, run_id, 10



def run_sim(run_id, include_noise, lsst):

    base_dir = (
        "lhc_run_lsst_sims" if include_noise 
        else "lhc_run_lsst_sim_zero_err"
    )

    # add LSST subfolder
    lsst_dir = os.path.join(base_dir, f"lsst_{lsst}")

    run_dir = os.path.join(lsst_dir, f"run_{run_id}")
    seed = run_id
    np.random.seed(seed)

    # first run path must follow same structure
    first_run_dir = os.path.join(base_dir, f"lsst_{lsst}", "run_1")
    first_shell = os.path.join(first_run_dir, "shell_cls.npy")

    # Create config
    config = Config(
        sim_dir=run_dir,
        seed=seed,
        include_shape_noise=include_noise,
        lsst=lsst,
    )
    config.save()

    print(f"\n=== include_shape_noise={include_noise} | lsst={lsst} | run {run_id} ===")

    # Step 1 logic
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

    step2(config)
    step3(config)

    print(f"Finished run {run_id} (include_noise={include_noise}, lsst={lsst})")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, required=True)
    args = parser.parse_args()

    include_noise, run_id, lsst = map_missing_job_to_config(args.job_id)
    run_sim(run_id, include_noise, lsst)

    
    

