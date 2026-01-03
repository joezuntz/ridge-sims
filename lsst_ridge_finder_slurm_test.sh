#!/bin/bash
#SBATCH --job-name=lsst_ridges
#SBATCH --output=logs/lsst_ridges_%A_%a.out
#SBATCH --error=logs/lsst_ridges_%A_%a.err

#SBATCH --nodes=1               
#SBATCH --ntasks=8           
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=10-00:00:00

source ~/.bashrc
conda activate /home/jzuntz/ridges/env
export OMP_NUM_THREADS=1
mpirun -n $SLURM_NTASKS python lsst_sims/lsst_ridges_slurm_test_sharedArray.py --task-id 1
