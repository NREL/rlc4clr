#!/bin/bash
#SBATCH --account=drl4dsr
#SBATCH --time=5:00:00
#SBATCH --job-name=kt
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --tasks-per-node=1


module purge
module load anaconda3
conda activate /projects/drl4dsr/xzhang2/conda-envs/rlc4clr

# 1. Generate training data
python kt_data_generation.py

# 2. Behavior cloning

# This script trains all 4 RL controllers that uses different
# lookahead length (1,2,4 and 6 hours). You can modify the code
# here to focus on one case and accelerate the training process.

for i in 1 2 4 6
do 
  python kt_behavior_clone.py --forecast-len $i
done
