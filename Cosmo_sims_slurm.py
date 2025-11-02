import os
import sys
import numpy as np
import argparse
from ridge_sims.steps import step1, step2, step3
from ridge_sims.config import Config

# =====================
# Global setup
# =====================

base_sim_dir = "lhc_cosmo_sims_zero_err"

# Fiducial values
Omega_m_fid = 0.32
S8_fid = 0.78
sigma8_fid = S8_fid / np.sqrt(Omega_m_fid / 0.3)

categories = ["S8", "S8_perp", "Om_fixed", "sigma8_fixed"]
num_per_category = 10


def get_parameters(category, run_idx):
    """Return Omega_m and sigma8 for a given category and run index."""
    Omega_m_range = np.linspace(0.24, 0.40, num_per_category)
    sigma8_range = np.linspace(0.66, 0.88, num_per_category)
    S8_range = np.linspace(0.65, 0.95, num_per_category)  # Even spacing in S8

    if category == "S8":
    
        Omega_m = Omega_m_range[run_idx]
        S8 = S8_range[run_idx]
        sigma8 = S8 / np.sqrt(Omega_m / 0.3)

    elif category == "S8_perp":
        # Constant S8, varying Omega_m
        Omega_m = Omega_m_range[run_idx]
        sigma8 = S8_fid * np.sqrt(0.3 / Omega_m)

    elif category == "Om_fixed":
        # Fixed Omega_m, varying sigma8
        Omega_m = Omega_m_fid
        sigma8 = sigma8_range[run_idx]

    elif category == "sigma8_fixed":
        # Fixed sigma8, varying Omega_m
        Omega_m = Omega_m_range[run_idx]
        sigma8 = sigma8_fid

    else:
        raise ValueError(f"Unknown category: {category}")

    return Omega_m, sigma8


def run_cosmology(category, run_idx):
    """Run one cosmology given the category and index."""
    Omega_m, sigma8 = get_parameters(category, run_idx)
    S8 = sigma8 * np.sqrt(Omega_m / 0.3)

    run_id = f"run_{run_idx + 1}"
    sim_dir = os.path.join(base_sim_dir, category, run_id)
    os.makedirs(sim_dir, exist_ok=True)

    config = Config(Omega_m=Omega_m, sigma8=sigma8, sim_dir=sim_dir, seed=run_idx, include_shape_noise = False) # make no noise
    config.save()

    print(f"Starting cosmology {category}/{run_id}")
    print(f"  Omega_m={Omega_m:.4f}, sigma8={sigma8:.4f}, S8={S8:.4f}")
    print(f"  Directory: {sim_dir}")
    sys.stdout.flush()

    step1(config)
    print("  -> step1 complete"); sys.stdout.flush()
    step2(config)
    print("  -> step2 complete"); sys.stdout.flush()
    step3(config)
    print("  -> step3 complete"); sys.stdout.flush()

    print(f"Finished run {category}/{run_id}")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, required=True, help="SLURM array job ID (0–39)")
    args = parser.parse_args()

    num_categories = len(categories)
    total_jobs = num_categories * num_per_category

    if not (0 <= args.job_id < total_jobs):
        raise ValueError(f"job-id must be between 0 and {total_jobs-1}")

    # Determine which category and run index this job corresponds to
    category_idx = args.job_id // num_per_category
    run_idx = args.job_id % num_per_category

    category = categories[category_idx]
    run_cosmology(category, run_idx)
