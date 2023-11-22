#!/bin/bash
#SBATCH --account=drl4dsr
#SBATCH --time=2:00:00
#SBATCH --job-name=kt
#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --tasks-per-node=1


module purge
module load anaconda3
conda activate /projects/drl4dsr/xzhang2/conda-envs/rlc4clr

python kt_behavior_clone.py --forecast-len 2

