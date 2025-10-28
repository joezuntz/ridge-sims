#!/bin/bash
#SBATCH --job-name=lhc_cosmo_sims
#SBATCH --output=logs/lhc_cosmo_sims_%A_%a.out
#SBATCH --error=logs/lhc_cosmo_sims_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G


source ~/.bashrc
conda activate /home/jzuntz/ridges/env
export OMP_NUM_THREADS=32

# Mirror Python logic for category assignment
categories=("S8" "S8_perp" "Om_fixed" "sigma8_fixed")
num_per_category=10
category_idx=$((SLURM_ARRAY_TASK_ID / num_per_category))
category=${categories[$category_idx]}

echo "Running category: $category (job ID: $SLURM_ARRAY_TASK_ID)"
python Cosmo_sims_slurm.py --job-id=$SLURM_ARRAY_TASK_ID
