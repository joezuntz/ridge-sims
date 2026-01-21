#!/bin/bash
#SBATCH --job-name=des_fiducial
#SBATCH --output=logs/des_fiducial_%j.out
#SBATCH --error=logs/des_fiducial_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

# =====================
# Environment
# =====================
source ~/.bashrc
conda activate /home/jzuntz/ridges/env

export OMP_NUM_THREADS=32

# =====================
# Run
# =====================
echo "Running DES fiducial simulation"
python run_fiducial_DES_sim_slurm.py
