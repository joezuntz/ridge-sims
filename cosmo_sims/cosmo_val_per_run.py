import numpy as np
import pandas as pd

Omega_m_fid = 0.32
S8_fid = 0.78
sigma8_fid = S8_fid / np.sqrt(Omega_m_fid / 0.3)

categories = ["S8", "S8_perp", "Om_fixed", "sigma8_fixed"]
num = 10

Omega_m_range = np.linspace(0.24, 0.40, num)
sigma8_range = np.linspace(0.66, 0.88, num)
S8_range = np.linspace(0.65, 0.95, num)

rows = []

for category in categories:
    for i in range(num):
        run = f"run_{i+1}"

        if category == "S8":
            Omega_m = Omega_m_fid
            S8 = S8_range[i]
            sigma8 = S8 / np.sqrt(Omega_m / 0.3)

        elif category == "S8_perp":
            Omega_m = Omega_m_range[i]
            S8 = S8_fid
            sigma8 = S8_fid / np.sqrt(Omega_m / 0.3)

        elif category == "Om_fixed":
            Omega_m = Omega_m_fid
            sigma8 = sigma8_range[i]
            S8 = sigma8 * np.sqrt(Omega_m / 0.3)

        elif category == "sigma8_fixed":
            Omega_m = Omega_m_range[i]
            sigma8 = sigma8_fid
            S8 = sigma8 * np.sqrt(Omega_m / 0.3)

        rows.append(dict(
            category=category,
            run=run,
            Omega_m=Omega_m,
            sigma8=sigma8,
            S8=S8
        ))

df = pd.DataFrame(rows)
df.to_csv("cosmo_run_mapping.csv", index=False)

print("Saved cosmo_run_mapping.csv")
print(df)