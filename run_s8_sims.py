# run_s8_sims.py

from ridge_sims.steps import step1, step2, step3
from ridge_sims.config import Config


if __name__ == "__main__":

    for i in range(10):
        sigma8 = 0.8 + (i-5) * 0.01  # CHANGEME
        Omega_m = 0.3 + (i-5) * 0.01 # CHANGEME
        
        config = Config(
            Omega_m=Omega_m,
            sigma8=sigma8,
            sim_dir="sim-i" #CHANGEME
            seed=i,
        )
        config.save()

        if i > 0:
            # copy the shell_cl file into the new directory
            ...

        step1(config)
        step2(config)
        step3(config)
