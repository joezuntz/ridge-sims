"""
We consider a hybrid configuration in which DES-like noise, sampling, and survey geometry are retained,
while the redshift distributions are replaced by those expected for LSST Year 10, therfore,
isolating the impact of redshift reach from that of number density and shape noise.

"""





import argparse
import os
import numpy as np
from ridge_sims.steps import step1, step2, step3
from ridge_sims.config import Config


def map_missing_job_to_config(job_id):
    """
    simulations for Des with lsst10 nz.
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

    return include_noise, run_id


def run_sim(run_id, include_noise):

    # OUTPUT 
    base_dir = "lhc_run_hybrid_DES_lSST10_sims"

    # Keep noise and zero-noise separated but still under the requested root
    noise_tag = "noise" if include_noise else "zero_err"
    sim_root = os.path.join(base_dir, noise_tag)

    #LSST MODE 
    lsst = 10

    # add LSST subfolder
    lsst_dir = os.path.join(sim_root, f"lsst10_nz")

    run_dir = os.path.join(lsst_dir, f"run_{run_id}")
    seed = run_id
    np.random.seed(seed)

    # first run path must follow same structure
    first_run_dir = os.path.join(lsst_dir, "run_1")
    first_shell = os.path.join(first_run_dir, "shell_cls.npy")

    # Create config
    config = Config(
        sim_dir=run_dir,
        seed=seed,
        include_shape_noise=include_noise,
        lsst=None,
        lsst10_nz=True,  # Turns on the hybrid mode
    )
    config.save()

    print(f"\n=== include_shape_noise={include_noise} | mode=DES+LSST10_nz | run {run_id} ===")

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

    include_noise, run_id = map_missing_job_to_config(args.job_id)
    run_sim(run_id, include_noise)
