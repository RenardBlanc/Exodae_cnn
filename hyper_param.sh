#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100l:4
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --mem=256T
#SBATCH --time=3:00
#SBATCH --account=def-fellouah
#SBATCH --mail-user=baktacheilyas@gmail.com 

module load cuda cudnn
# Activation du virtualenv
source env/bin/activate

python3 experience/exp_class.py 0 50000

