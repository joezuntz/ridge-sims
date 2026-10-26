import os
import sys
import numpy as np
from ridge_sims.steps import step1, step2, step3
from ridge_sims.config import Config

# =====================
# Global setup
# =====================

# single DES fiducial run
base_sim_dir = "fisher_runs"

# Fiducial cosmology values
Omega_m_fid = 0.3
sigma_8_fid = 0.8
Omega_m_values = [0.298, 0.299, 0.30, 0.301, 0.302]
sigma_8_values = [0.798, 0.799, 0.8, 0.801, 0.802]

def run_fisher_point(param, i):
    if param == "Omega_m":
        Omega_m = Omega_m_values[i]
        sigma8 = sigma_8_fid
    elif param == "sigma8":
        Omega_m = Omega_m_fid
        sigma8 = sigma_8_values[i]
    else:
        raise ValueError("Unknown param")

    S8 = sigma8 * np.sqrt(Omega_m / 0.3)

    run_id = f"fisher_run_{param}_{i}"
    sim_dir = os.path.join(base_sim_dir, run_id)
    os.makedirs(sim_dir, exist_ok=True)

    # Include shape noise
    config = Config(
        Omega_m=Omega_m,
        sigma8=sigma8,
        sim_dir=sim_dir,
        seed=0,
        include_shape_noise=False,  
    )
    config.save()

    print(f"Starting run: {run_id}")
    print(f"  Omega_m={Omega_m:.4f}, sigma8={sigma8:.4f}, S8={S8:.4f}")
    print(f"  Directory: {sim_dir}")
    sys.stdout.flush()

    step1(config); print("  -> step1 complete"); sys.stdout.flush()
    step2(config); print("  -> step2 complete"); sys.stdout.flush()
    step3(config, sigma_e_default=0.0); print("  -> step3 complete"); sys.stdout.flush()

    print(f"Finished run: {run_id}")
    sys.stdout.flush()

if __name__ == "__main__":
    for i in range(5):
        run_fisher_point("Omega_m")
        run_fisher_point("sigma8")
