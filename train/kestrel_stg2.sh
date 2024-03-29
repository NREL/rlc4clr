#!/bin/bash
#SBATCH --account=drl4dsr
#SBATCH --time=20:00:00
#SBATCH --job-name=kt
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --tasks-per-node=1


module purge
module load anaconda3
conda activate /projects/drl4dsr/xzhang2/conda-envs/rlc4clr

python train_stg2.py --forecast-len 6 --error-level 0.2
