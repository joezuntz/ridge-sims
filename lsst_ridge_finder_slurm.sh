#!/bin/bash
#SBATCH --job-name=lsst_ridges
#SBATCH --output=logs/lsst_ridges_%A_%a.out
#SBATCH --error=logs/lsst_ridges_%A_%a.err

#SBATCH --array=1-7
#SBATCH --nodes=1               
#SBATCH --ntasks=16             # 16 MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --mem=1024G
#SBATCH --time=10-00:00:00

source ~/.bashrc
conda activate /home/jzuntz/ridges/env
export OMP_NUM_THREADS=1
mpirun -n $SLURM_NTASKS python lsst_sims/lsst_ridges_slurm.py --task-id $SLURM_ARRAY_TASK_ID
