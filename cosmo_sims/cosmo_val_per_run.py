import os, sys

# Directory of this script (cosmo_sims)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up 
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# find modules in the parent directory
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# we stay inside cosmo_sims for outputs
os.chdir(current_dir)


import yaml
import numpy as np
import pandas as pd 

base_sim_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "lhc_cosmo_sims_zero_err"))

missing_runs = {}   # list of missing/broken runs

rows = []           # save
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


print("Mapping started")

# Loop over categories
for category in sorted(os.listdir(base_sim_dir)):
    cat_dir = os.path.join(base_sim_dir, category)
    if not os.path.isdir(cat_dir):
        continue

    print(f"\n-- {category} --")
    missing_runs[category] = []

    # Loop over runs 
    for run in sorted(os.listdir(cat_dir)):
        run_dir = os.path.join(cat_dir, run)

        if not run.startswith("run_") or not os.path.isdir(run_dir):
            continue

        yaml_file = os.path.join(run_dir, "config.yaml")
        if not os.path.exists(yaml_file):
            missing_runs[category].append(run)
            print(f"{run:8s}  →  MISSING config.yaml")

            rows.append({      
                "category": category,
                "run": run,
                "Omega_m": None,
                "sigma8": None,
                "S8": None,
                "status": "MISSING"
            })
            continue

        # Try reading YAML
        try:
            cfg = load_yaml(yaml_file)
        except Exception as e:
            missing_runs[category].append(run)
            print(f"{run:8s}  →  ERROR reading YAML ({e})")

            rows.append({     
                "category": category,
                "run": run,
                "Omega_m": None,
                "sigma8": None,
                "S8": None,
                "status": "BROKEN"
            })
            continue

        # Extract parameters
        try:
            Omega_m = cfg.get("Omega_m", None)
            sigma8  = cfg.get("sigma8", None)
        except Exception as e:
            missing_runs[category].append(run)
            print(f"{run:8s}  →  INVALID YAML FORMAT ({e})")

            rows.append({     
                "category": category,
                "run": run,
                "Omega_m": None,
                "sigma8": None,
                "S8": None,
                "status": "INVALID_FORMAT"
            })
            continue

        # Check missing parameters
        if Omega_m is None or sigma8 is None:
            missing_runs[category].append(run)
            print(f"{run:8s}  →  INVALID CONFIG (missing Omega_m or sigma8)")

            rows.append({     
                "category": category,
                "run": run,
                "Omega_m": Omega_m,
                "sigma8": sigma8,
                "S8": None,
                "status": "INVALID_CONFIG"
            })
            continue

        # Compute S8
        S8 = sigma8 * np.sqrt(Omega_m / 0.3)

        print(f"{run:8s}  →  Omega_m={Omega_m:.4f}   sigma8={sigma8:.4f}   S8={S8:.4f}")

        rows.append({       
            "category": category,
            "run": run,
            "Omega_m": Omega_m,
            "sigma8": sigma8,
            "S8": S8,
            "status": "OK"
        })


# Summary of missing runs

print("\n\n===== SUMMARY OF MISSING OR BROKEN RUNS =====")
any_missing = False

for category, runs in missing_runs.items():
    if len(runs) > 0:
        any_missing = True
        print(f"\n{category}:")
        for r in runs:
            print(f"   - {r}")

if not any_missing:
    print("\nNo missing runs — all good!")


# =====================================================
df = pd.DataFrame(rows)
df.to_csv("cosmo_run_mapping.csv", index=False)

print("Saved CSV: cosmo_run_mapping.csv")
