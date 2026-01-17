#!/bin/bash
#SBATCH --job-name=lsst_sims
#SBATCH --output=logs/lsst_sims_%A_%a.out
#SBATCH --error=logs/lsst_sims_%A_%a.err
#SBATCH --array=1-4
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G

source ~/.bashrc
conda activate /home/jzuntz/ridges/env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "SLURM job array ID = $SLURM_ARRAY_TASK_ID"

python run_hybrid_DES_lsst_sim_slurm.py --job-id $SLURM_ARRAY_TASK_ID

