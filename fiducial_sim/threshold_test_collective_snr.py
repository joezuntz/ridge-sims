import os
import glob
import numpy as np
from scipy import stats

# ----------------------------
# Path setup
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))
os.chdir(current_dir)

# threshold outputs:
ROOT = os.path.join(current_dir, "density_threshold_test", "shear_vs_threshold")


THRESHOLDS = [10, 15, 25, 40, 55, 70]
RUN_ID = 1

# ----------------------------
# Cov regularization 
# ----------------------------
def inv_cov(cov, eps=1e-12):
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        print("[WARN] Covariance singular -> adding diagonal regularization.")
        cov_reg = cov + np.eye(cov.shape[0]) * eps
        return np.linalg.inv(cov_reg)

def run_analysis(case_label, shear_csv, noise_files):
    # --- Load signal ---
    signal_data = np.loadtxt(shear_csv, delimiter=",", skiprows=1)
    g_plus_signal  = signal_data[:, 2]
    g_cross_signal = signal_data[:, 3]

    # --- Load noise realizations ---
    all_g_plus_noise = []
    all_g_cross_noise = []
    used_noise = 0

    for nf in noise_files:
        if not os.path.exists(nf):
            continue
        data = np.loadtxt(nf, delimiter=",", skiprows=1)
        all_g_plus_noise.append(data[:, 2])
        all_g_cross_noise.append(data[:, 3])
        used_noise += 1

    if used_noise < 2:
        raise RuntimeError(f"{case_label}: need >=2 noise realizations, found {used_noise}")

    all_g_plus_noise  = np.array(all_g_plus_noise)
    all_g_cross_noise = np.array(all_g_cross_noise)

    # --- Covariance matrices from noise realizations ---
    cov_plus  = np.cov(all_g_plus_noise,  rowvar=False, ddof=1)
    cov_cross = np.cov(all_g_cross_noise, rowvar=False, ddof=1)

    cov_plus_inv  = inv_cov(cov_plus)
    cov_cross_inv = inv_cov(cov_cross)

    # --- Hartlap factor (apply only if valid) ---
    N = used_noise
    p = cov_plus.shape[0]

    hartlap_applied = False
    hartlap = 1.0
    if N > p + 2:
        hartlap = (N - p - 2) / (N - 1)
        cov_plus_inv  = hartlap * cov_plus_inv
        cov_cross_inv = hartlap * cov_cross_inv
        hartlap_applied = True

    # --- Chi-square ---
    d_plus  = g_plus_signal
    d_cross = g_cross_signal
    dof = len(d_plus)

    chi2_plus  = float(d_plus  @ cov_plus_inv  @ d_plus)
    chi2_cross = float(d_cross @ cov_cross_inv @ d_cross)

    # p-values
    pval_plus  = stats.chi2.sf(chi2_plus,  dof)
    pval_cross = stats.chi2.sf(chi2_cross, dof)

    # "SNR": sqrt(chi2)
    snr_plus  = np.sqrt(chi2_plus)  if chi2_plus  >= 0 else np.nan
    snr_cross = np.sqrt(chi2_cross) if chi2_cross >= 0 else np.nan

    return {
        "label": case_label,
        "Nrand": N,
        "p": p,
        "hartlap": hartlap,
        "hartlap_applied": hartlap_applied,
        "dof": dof,
        "chi2_plus": chi2_plus,
        "chi2_cross": chi2_cross,
        "chi2dof_plus": chi2_plus / dof,
        "chi2dof_cross": chi2_cross / dof,
        "pval_plus": pval_plus,
        "pval_cross": pval_cross,
        "snr_plus": snr_plus,
        "snr_cross": snr_cross,
    }

def collect_case_paths(root, run_id, thr):
    shear_csv = os.path.join(root, f"run_{run_id}_shear_fp_{thr}.csv")
    noise_dir = os.path.join(root, f"run_{run_id}_random_shear_p{thr}")

    if not os.path.exists(shear_csv):
        raise FileNotFoundError(f"Missing signal CSV: {shear_csv}")
    if not os.path.isdir(noise_dir):
        raise FileNotFoundError(f"Missing noise dir: {noise_dir}")

    noise_files = sorted(glob.glob(os.path.join(noise_dir, "*.csv")))
    if len(noise_files) == 0:
        raise FileNotFoundError(f"No noise CSVs found in: {noise_dir}")

    return shear_csv, noise_files

if __name__ == "__main__":

    results = []
    for thr in THRESHOLDS:
        shear_csv, noise_files = collect_case_paths(ROOT, RUN_ID, thr)
        res = run_analysis(
            case_label=f"fp={thr}",
            shear_csv=shear_csv,
            noise_files=noise_files,
        )
        results.append(res)

    # Print a table
    print("\n=== Chi2 / SNR by threshold ===")
    header = (
        "fp  Nrand  p  Hartlap  dof   "
        "chi2+   chi2+/dof   SNR+    pval+     "
        "chi2x   chi2x/dof   SNRx    pvalx"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        fp = int(r["label"].split("=")[1])
        hart = r["hartlap"] if r["hartlap_applied"] else np.nan
        print(
            f"{fp:>2d}  "
            f"{r['Nrand']:>5d}  "
            f"{r['p']:>2d}  "
            f"{hart:>7.4f}  "
            f"{r['dof']:>3d}   "
            f"{r['chi2_plus']:>7.3f}  "
            f"{r['chi2dof_plus']:>10.3f}  "
            f"{r['snr_plus']:>6.2f}  "
            f"{r['pval_plus']:>8.2e}   "
            f"{r['chi2_cross']:>7.3f}  "
            f"{r['chi2dof_cross']:>10.3f}  "
            f"{r['snr_plus']:>6.2f}  "
            f"{r['pval_plus']:>8.2e}"
        )

    # save CSV summary 
    out_csv = os.path.join(current_dir, "chi2_snr_by_threshold.csv")
    with open(out_csv, "w") as f:
        f.write("fp,Nrand,p,hartlap_applied,hartlap,dof,chi2_plus,chi2dof_plus,snr_plus,pval_plus,chi2_cross,chi2dof_cross,snr_cross,pval_cross\n")
        for r in results:
            fp = int(r["label"].split("=")[1])
            f.write(
                f"{fp},{r['Nrand']},{r['p']},{int(r['hartlap_applied'])},{r['hartlap']},"
                f"{r['dof']},{r['chi2_plus']},{r['chi2dof_plus']},{r['snr_plus']},{r['pval_plus']},"
                f"{r['chi2_cross']},{r['chi2dof_cross']},{r['snr_cross']},{r['pval_cross']}\n"
            )

    print(f"\n[OK] Wrote summary CSV: {out_csv}")