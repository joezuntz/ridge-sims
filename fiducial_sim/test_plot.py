import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# PATHS
# -------------------------------------------------
#current_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))

#ridge_file = os.path.join(
#    current_dir,
#    "DES_fiducial_sim/band_0.1_mesh_2/Ridges_final_p15/shape_err_run_1_ridges_p15_contracted.h5"
#)

#bg_file = os.path.join(
#    parent_dir,
#    "lhc_DES_fiducial_sim",
#    "run_1",
#    "source_catalog_cutzl0.40.h5"
#)

#out_png = "sim_background_with_ridges.png"

# -------------------------------------------------
# LOAD BACKGROUND 
# -------------------------------------------------
#with h5py.File(bg_file, "r") as f:
#    ra_bg  = f["RA"][:]        # degrees
#    dec_bg = f["DEC"][:]       # degrees

#ra_bg = (ra_bg + 180.0) % 360.0   # REQUIRED SHIFT

# -------------------------------------------------
# LOAD RIDGES (radians → degrees)
# -------------------------------------------------
#with h5py.File(ridge_file, "r") as f:
#    ridges = f["ridges"][:]

#dec_r = np.degrees(ridges[:, 0])
#ra_r  = np.degrees(ridges[:, 1])

# -------------------------------------------------
#  NUMERICAL CHECK 
# -------------------------------------------------
#print("[CHECK] Background RA range :", ra_bg.min(), ra_bg.max())
#print("[CHECK] Background DEC range:", dec_bg.min(), dec_bg.max())
#print("[CHECK] Ridges RA range     :", ra_r.min(), ra_r.max())
#print("[CHECK] Ridges DEC range    :", dec_r.min(), dec_r.max())

# -------------------------------------------------
# PLOT
# -------------------------------------------------
#plt.figure(figsize=(10, 6.5))

#H, xedges, yedges = np.histogram2d(
#    ra_bg, dec_bg,
#    bins=[1200, 800]
#)
#H = np.log10(H + 1)

#plt.imshow(
#    H.T,
#    origin="lower",
#    aspect="auto",
#    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
#    cmap="Greys"
#)

#plt.scatter(
#    ra_r, dec_r,
#    s=0.2,
#    c="red",
#    alpha=0.9,
#    label="Contracted ridges"
#)

#plt.xlabel("RA [deg]")
#plt.ylabel("DEC [deg]")
#plt.title("Simulation background + contracted ridges (shifted)")
#plt.legend(markerscale=10)
#plt.tight_layout()
#plt.savefig(out_png, dpi=250)
#plt.close()

#print(f"[OK] Saved → {out_png}")



current_dir = os.path.dirname(os.path.abspath(__file__))

#csv_file = os.path.join(current_dir, "shear/DES_fiducial_sim/band_0.1_mesh_2/run_1_p15/signal_shear.csv")
csv_file = os.path.join(current_dir, "shear_no_endpoint/signal_shear.csv")
out_png  = os.path.join(current_dir, "shear_no_endpoint", "gamma_plus.png")

if not os.path.exists(csv_file):
    raise FileNotFoundError(csv_file)

# load (skip header row)
d = np.loadtxt(csv_file, delimiter=",", skiprows=1)

bin_center = d[:, 0]          # radians
real_dist  = d[:, 1]          # radians (weighted)
g_plus     = d[:, 2]

# plot vs distance in arcmin (more interpretable)
x_arcmin = np.degrees(real_dist) * 60.0

plt.figure(figsize=(7.5, 6.0))
plt.plot(x_arcmin, g_plus, marker="o", lw=1.5, ms=4)
plt.xscale("log")
plt.xlabel("Angular separation [arcmin]")
plt.ylabel(r"$\gamma_{+}$")
plt.title(r"shear profile $\gamma_{+}$")
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig(out_png, dpi=250)
plt.close()

print(f"[OK] Saved → {out_png}")
