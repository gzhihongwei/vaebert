#!/bin/bash

#SBATCH -J VAE-ShapeNet
#SBATCH -o output/%j.log
#SBATCH -e output/%j.err
#SBATCH -c 2
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 12:00:00

module load cuda/11.3.1

conda activate vaebert
python3 train.py
conda deactivate